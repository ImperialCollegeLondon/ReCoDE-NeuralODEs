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
    tableau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """

    Given a callable function that operates on two tensors, the state and the time, computes the
    numerical integration following the procedure specified by tableau.

    :param fn: The derivative of the system
    :param c_state: The current state of the system
    :param c_time: The current time of the system
    :param final_time: The time to integrate up to
    :param dt: The timestep of the integration
    :param tableau: The Butcher Tableau specifying the integration scheme
    :return: tuple of the current state and time after integration, and list of intermediate states
    """
    # We'd like to track the whole trajectory, not just the final state, so we use a list to store the intermediate states and times
    i_states = [(c_state.clone(), c_time.clone())]

    c_state, truncated_bits_state = util.partial_compensated_sum(c_state)
    c_time, truncated_bits_time = util.partial_compensated_sum(c_time)

    # We integrate as long as the current_time+dt is less than t1
    # Essentially, we want to stop the integration if taking another step
    # would lead to exceeding the final time
    while torch.any((c_time + dt) < final_time):
        # To compute the change in state, we add the truncated bits back
        delta_state, delta_time = helpers.compute_step(fn, c_state, c_time, dt, tableau)
        c_state, truncated_bits_state = util.partial_compensated_sum(
            delta_state, (c_state, truncated_bits_state)
        )
        c_time, truncated_bits_time = util.partial_compensated_sum(
            delta_time, (c_time, truncated_bits_time)
        )
        # We would like to store the intermediate states with the compensated partial sums
        i_states.append((c_state + truncated_bits_state, c_time + truncated_bits_time))

    delta_state, delta_time = helpers.compute_step(
        fn,
        c_state + truncated_bits_state,
        c_time + truncated_bits_time,
        final_time - (c_time + truncated_bits_time),
        tableau,
    )
    c_state, truncated_bits_state = util.partial_compensated_sum(
        delta_state, (c_state, truncated_bits_state)
    )
    c_time, truncated_bits_time = util.partial_compensated_sum(
        delta_time, (c_time, truncated_bits_time)
    )

    i_states.append((c_state + truncated_bits_state, c_time + truncated_bits_time))

    return c_state, c_time, i_states
