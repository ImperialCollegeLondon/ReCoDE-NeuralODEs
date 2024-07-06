import torch
import typing

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
    while torch.any(
        torch.where(dt > 0.0, (c_time + dt) < final_time, (c_time + dt) > final_time)
    ):
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

        non_finite_step = torch.any(~torch.isfinite(delta_state))
        # Euler estimate of the step, first stage of any RK method is the basic Euler estimate
        # which we use as a reference to check the stability of the step
        linear_variation_magnitude = 5.0 * torch.max(torch.abs(intermediate_stages[0]))
        step_variation_too_large = (
            torch.max(torch.abs(delta_state)) > linear_variation_magnitude
        )
        invalid_step = non_finite_step | step_variation_too_large
        redo_step = invalid_step

        if is_adaptive and not invalid_step:
            # We use `torch.linalg.norm` to compute the magnitude of the error
            # we can adjust this by passing in the `ord` keyword to choose a different
            # vector norm, but the 2-norm suffices for our purposes
            error_in_state.append((delta_state_upper - delta_state_lower).abs().max())
            with torch.no_grad():
                max_error = atol + torch.linalg.norm(rtol * c_state)
                dt, error_too_large = helpers.adapt_adaptive_timestep(
                    dt,
                    error_in_state[-1],
                    max_error,
                    integrator_order,
                    min_timestep=min_timestep,
                    max_timestep=max_timestep,
                )
                redo_step = redo_step | error_too_large
        else:
            error_in_state.append(error_in_state[0])
        if redo_step and trial_restarts < 8:
            if invalid_step and trial_restarts <= 32:
                intermediate_stages.fill_(0.0)
                dt = torch.copysign(
                    torch.clamp(torch.abs(dt) / 2, min=min_timestep, max=max_timestep),
                    dt,
                )
            # If the error exceeds our error threshold, we don't commit the step and redo it
            error_in_state = error_in_state[:-1]
            trial_restarts += 1
            continue
        else:
            if trial_restarts == 32:
                raise RuntimeError(
                    f"Encountered convergence failure at t={c_time} with timestep={dt}"
                )
            trial_restarts = 0

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
