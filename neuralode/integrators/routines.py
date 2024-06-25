import torch
import typing
import warnings

from neuralode.integrators import helpers
from neuralode import util


def solve_ivp(
    forward_fn: typing.Callable[[torch.Tensor, torch.Tensor, typing.Any], torch.Tensor],
    integrator_specification: tuple[torch.Tensor, bool, int, float, bool],
    x0: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    dt: torch.Tensor,
    atol: torch.Tensor,
    rtol: torch.Tensor,
    additional_dynamic_args=tuple(),
) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A general integration routine for solving an Initial Value Problem
    using any arbitrary Butcher Tableau

    :param forward_fn: The function to be integrated
    :param x0: the initial state to integrate from
    :param t0: the initial time to integrate from
    :param t1: the final time to integrate to
    :param dt: the time increments to integrate with
    :param atol: The absolute tolerance for the error in an adaptive integration
    :param rtol: The relative tolerance for the error in an adaptive integration
    :param additional_dynamic_args: additional arguments to pass to the function
    :return: A tuple of the final state (torch.Tensor), the final time (torch.Tensor),
             the intermediate states (torch.Tensor), the intermediate times (torch.Tensor),
             the error values (torch.Tensor)
    """

    (
        tableau,
        is_adaptive,
        number_of_stages,
        integrator_order,
        use_local_extrapolation,
    ) = integrator_specification

    if is_adaptive:
        atol = helpers.ensure_tolerance(atol, x0)
        rtol = helpers.ensure_tolerance(rtol, x0)

    dt = helpers.ensure_timestep(t0, t1, dt).detach()

    min_step_size = (util.next_value(t0, dt) - t0).detach()

    c_state = x0.clone()
    c_time = t0.clone()

    # Instead of storing each (x,t) pair in a list, we'll store them in separate tensors
    intermediate_states = [c_state]
    intermediate_times = [c_time]

    error_in_state = [torch.zeros(tuple(), device=x0.device, dtype=x0.dtype)]

    # We need to store the intermediate stages
    k_stages = torch.stack([torch.zeros_like(c_state)] * number_of_stages)

    c_state, truncated_bits_state = util.partial_compensated_sum(c_state)
    c_time, truncated_bits_time = util.partial_compensated_sum(c_time)

    step_args = dict(
        additional_dynamic_args=additional_dynamic_args,
        intermediate_stages=k_stages,
        is_adaptive=is_adaptive,
    )

    while torch.any(torch.where(t1 > t0, (c_time + dt) < t1, (c_time - dt) > t1)):
        delta_state_lower, delta_state_upper, delta_time = helpers.compute_step(
            forward_fn,
            c_state + truncated_bits_state,
            c_time + truncated_bits_time,
            dt,
            tableau,
            **step_args,
        )
        # If local extrapolation is enabled, we take the higher order estimate, otherwise the lower order one
        delta_state = (
            delta_state_upper if use_local_extrapolation else delta_state_lower
        )

        # We use `torch.linalg.norm` to compute the magnitude of the error
        # we can adjust this by passing in the `ord` keyword to choose a different
        # vector norm, but the 2-norm suffices for our purposes
        error_in_state.append(torch.linalg.norm(delta_state_upper - delta_state_lower))
        if is_adaptive:
            # To save on computation, we only compute the max error tolerated and the step
            # correction when the method is adaptive
            max_error = atol + torch.linalg.norm(rtol * c_state)
            step_correction = 0.8 * torch.where(
                error_in_state[-1] != 0.0, max_error / error_in_state[-1], 1.0
            ) ** (1 / integrator_order)
            # Based on the error, we correct the step size
            dt = torch.copysign(
                torch.maximum(
                    torch.abs(min_step_size), torch.abs(step_correction.detach() * dt)
                ),
                dt,
            )
            if error_in_state[-1].detach() >= max_error.detach():
                # If the error exceeds our error threshold, we don't commit the step and redo it
                error_in_state = error_in_state[:-1]
                continue
        c_state, truncated_bits_state = util.partial_compensated_sum(
            delta_state, (c_state, truncated_bits_state)
        )
        c_time, truncated_bits_time = util.partial_compensated_sum(
            delta_time, (c_time, truncated_bits_time)
        )

        intermediate_states.append(c_state + truncated_bits_state)
        intermediate_times.append(c_time + truncated_bits_time)

    delta_state_lower, delta_state_upper, delta_time = helpers.compute_step(
        forward_fn,
        c_state + truncated_bits_state,
        c_time + truncated_bits_time,
        (t1 - c_time) - truncated_bits_time,
        tableau,
        **step_args,
    )
    delta_state = delta_state_upper if use_local_extrapolation else delta_state_lower
    c_state, truncated_bits_state = util.partial_compensated_sum(
        delta_state, (c_state, truncated_bits_state)
    )
    c_state = c_state + truncated_bits_state
    c_time, truncated_bits_time = util.partial_compensated_sum(
        delta_time, (c_time, truncated_bits_time)
    )
    c_time = c_time + truncated_bits_time

    error_in_state.append(torch.linalg.norm(delta_state_upper - delta_state_lower))

    intermediate_states.append(c_state)
    intermediate_times.append(c_time)

    # As we said, these need to be converted to tensors for proper tracking
    intermediate_states = torch.stack(intermediate_states, dim=0)
    intermediate_times = torch.stack(intermediate_times, dim=0)

    # We should also put the errors we're returning into a tensor too
    error_in_state = torch.stack(error_in_state, dim=0)

    return c_state, c_time, intermediate_states, intermediate_times, error_in_state
