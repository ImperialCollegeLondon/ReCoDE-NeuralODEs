import torch
import typing
import numpy
import warnings

from neuralode import util
from neuralode.integrators import helpers


def integrate_system(
        fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        c_state: torch.Tensor,
        c_time: torch.Tensor,
        final_time: torch.Tensor,
        dt: torch.Tensor,
        integrator_kwargs,
):
    """
    A general integration routine for solving an Initial Value Problem
    using any arbitrary Butcher Tableau

    :param fn: The function to be integrated
    :param c_state: the initial state to integrate from
    :param c_time: the initial time to integrate from
    :param final_time: the final time to integrate to
    :param dt: the time increments to integrate with
    :param integrator_kwargs: Additional arguments for the integrator
    :return: A tuple of the final state (torch.Tensor), the final time (torch.Tensor),
             the intermediate states (torch.Tensor), the intermediate times (torch.Tensor),
             the error values (torch.Tensor)
    """
    if integrator_kwargs is None:
        integrator_kwargs = {}

    tableau = integrator_kwargs.get("tableau")
    is_adaptive = integrator_kwargs.get("is_adaptive")
    number_of_stages = integrator_kwargs.get("number_of_stages")
    integrator_order = integrator_kwargs.get("integrator_order")
    use_local_extrapolation = integrator_kwargs.get("use_local_extrapolation")

    if is_adaptive:
        atol = helpers.ensure_tolerance(
            integrator_kwargs.get("atol", torch.zeros_like(c_state)), c_state
        )
        rtol = helpers.ensure_tolerance(
            integrator_kwargs.get(
                "rtol", torch.ones_like(c_state) * torch.finfo(c_state.dtype).eps ** 0.5
            ),
            c_state,
        )

    min_timestep = integrator_kwargs.get("min_dt", None)
    if min_timestep is not None:
        min_timestep = torch.abs(min_timestep)

    max_timestep = integrator_kwargs.get("max_dt", None)
    if max_timestep is not None:
        max_timestep = torch.abs(max_timestep)

    dt = helpers.ensure_timestep(dt, c_time, final_time).detach()

    smallest_valid_timestep = 10 * torch.abs(util.next_value(c_time, dt) - c_time)
    if min_timestep is None:
        min_timestep = smallest_valid_timestep
    else:
        min_timestep = torch.maximum(smallest_valid_timestep, min_timestep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        min_timestep = helpers.ensure_timestep(min_timestep, c_time, final_time)
        if max_timestep is not None:
            max_timestep = helpers.ensure_timestep(max_timestep, c_time, final_time)
        if (final_time - c_time) < 0.0:
            min_timestep, max_timestep = max_timestep, min_timestep

    i_states = [(c_state.clone(), c_time.clone())]
    error_in_state = [torch.zeros(tuple(), device=c_state.device, dtype=c_state.dtype)]

    c_state, truncated_bits_state = util.partial_compensated_sum(c_state)
    c_time, truncated_bits_time = util.partial_compensated_sum(c_time)

    intermediate_stages = torch.zeros(
        (number_of_stages, *c_state.shape), dtype=c_state.dtype, device=c_state.device
    )

    common_step_arguments = (tableau, intermediate_stages, is_adaptive)

    # Sometimes a timestep will fail because it is too large
    # we need to track when the integrator gets stuck in a loop due to this
    trial_restarts = 0
    while torch.any(torch.where(dt > 0.0, (c_time + dt) < final_time, (c_time + dt) > final_time)):
        delta_state_lower, delta_state_upper, delta_time = helpers.compute_step(
            fn,
            c_state + truncated_bits_state,
            c_time + truncated_bits_time,
            dt,
            *common_step_arguments,
        )

        # If local extrapolation is enabled, we take the higher order estimate, otherwise the lower order one
        delta_state = (
            delta_state_upper if use_local_extrapolation else delta_state_lower
        )
        # Sometimes, the estimated delta is not finite because of divergence
        # either in the implementation of forward_fn or because of accumulated errors.
        # We raise an error when this happens to let the user know to debug the issue
        # and to reduce the tolerances on the errors if needed
        non_finite_step = torch.any(~torch.isfinite(delta_state))
        raise_error = non_finite_step

        with torch.no_grad():
            if is_adaptive:
                # Euler estimate of the step, first stage of any RK method is the basic Euler estimate
                # which we use as a reference to check the stability of the step
                # Similarly, divergence can occur much sooner than a non-finite step, and
                # to detect this we check that the step does not vary more than 25 times the basic Euler step.
                # In many cases where the Euler method does not diverge, this is a good test and if it does diverge
                # it will not give a false positive, but instead a false negative.
                linear_variation_magnitude = 25.0 * torch.max(torch.abs(intermediate_stages[0])).clamp(min=torch.finfo(c_state.dtype).eps**0.5).detach()
                step_variation_too_large = (
                        torch.max(torch.abs(delta_state.detach())) > linear_variation_magnitude
                )
                redo_step = non_finite_step | step_variation_too_large
                raise_error = redo_step & (trial_restarts > 8)

                # We only need to adapt the timestep if we haven't hit one of the issue conditions above
                if ~redo_step:
                    # We use the per-component error to adapt the timestep.
                    # In this fashion, we can ensure that the error on every component
                    # is within the tolerances rather than the vector norm.
                    current_error = (delta_state_upper - delta_state_lower).abs()
                    max_error = (atol + rtol * torch.maximum(c_state.abs(), (c_state + delta_state).abs()))
                    dt, error_too_large = helpers.adapt_adaptive_timestep(
                        dt,
                        current_error,
                        max_error,
                        integrator_order,
                        min_timestep=min_timestep,
                        max_timestep=max_timestep,
                    )
                    # We redo the step if the error is large and we have not been repeatedly
                    # attempting a step and failing.
                    redo_step = error_too_large & (trial_restarts <= 8)
                    # If the timestep is on the lower or upper bounds, this means that the
                    # error is due to a user restriction and we silently take the step regardless.
                    if min_timestep is not None:
                        redo_step &= (dt.abs() > min_timestep.abs())
                    if max_timestep is not None:
                        redo_step &= (dt.abs() < max_timestep.abs())
                    if ~redo_step:
                        error_in_state.append(current_error.max())
                else:
                    # If the step failed due to a non-finite step or having too large a variation,
                    # we can reduce the timestep and try again.
                    dt = torch.clamp(0.25 * dt, min=min_timestep, max=max_timestep)
                if not raise_error and redo_step:
                    trial_restarts += 1
                    continue
                trial_restarts = 0
        if raise_error:
            err = RuntimeError(
                f"Encountered convergence failure at t={c_time} with timestep={dt}"
            )
            err.add_note(
                f"""with the following state:
current time: {c_time}/timestep: {dt}
current state: {numpy.array2string(c_state.cpu().numpy(), separator=', ', precision=16)}
delta state: {numpy.array2string(delta_state.cpu().numpy(), separator=', ', precision=16)}
""")
            raise err

        c_state, truncated_bits_state = util.partial_compensated_sum(
            delta_state, (c_state, truncated_bits_state)
        )
        c_time, truncated_bits_time = util.partial_compensated_sum(
            delta_time, (c_time, truncated_bits_time)
        )

        i_states.append((c_state + truncated_bits_state, c_time + truncated_bits_time))

    delta_state_lower, delta_state_upper, delta_time = helpers.compute_step(
        fn,
        c_state + truncated_bits_state,
        c_time + truncated_bits_time,
        (final_time - c_time) - truncated_bits_time,
        *common_step_arguments,
    )
    delta_state = delta_state_upper if use_local_extrapolation else delta_state_lower

    c_state, truncated_bits_state = util.partial_compensated_sum(
        delta_state, (c_state, truncated_bits_state)
    )
    c_time, truncated_bits_time = util.partial_compensated_sum(
        delta_time, (c_time, truncated_bits_time)
    )

    error_in_state.append(torch.linalg.norm(delta_state_upper - delta_state_lower))
    i_states.append((c_state + truncated_bits_state, c_time + truncated_bits_time))

    return c_state, c_time, i_states, error_in_state
