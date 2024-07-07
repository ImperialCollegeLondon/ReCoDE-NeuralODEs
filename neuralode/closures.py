import torch
import neuralode.integrators


def dynamics_closure(rhs: neuralode.integrators.signatures.integration_fn_signature, parameters: list[torch.Tensor], minibatch: dict[str, torch.Tensor],
                     optimiser: torch.optim.Optimizer, integrator_kwargs: dict[str, torch.Tensor], integrator: neuralode.integrators.classes.Integrator=None) -> torch.Tensor:
    """
    Computes the error on a minibatch of states and times according to the given function `rhs` and its parameters.

    Given a minibatch containing pairs of states and times, this function computes the error at each time between the parameterised dynamics
    and the reference states. Useful for learning the dynamics of a given system.

    :param rhs: The function to integrate
    :param parameters: The parameters wrt. which the gradients should be computed
    :param minibatch: The minibatch to compute the error on
    :return: The error of the integration on the minibatch
    """
    if integrator is None:
        integrator = neuralode.integrators.AdaptiveRK45Integrator

    current_state = minibatch['initial_state'].detach().clone()
    current_time = minibatch['initial_time'].detach().clone()
    initial_timestep = minibatch['dt'].detach().clone()

    times = minibatch['times']
    states = minibatch['states']

    # We need to sort both times and states simultaneously, so we'll use `argsort`
    sorted_time_indices = torch.argsort(times)
    times, states = times[sorted_time_indices], states[sorted_time_indices]

    optimiser.zero_grad()
    error = 0.0

    for sample_state, sample_time in zip(states, times):
        dt = torch.minimum(initial_timestep, sample_time - current_time).detach()
        current_state, current_time, _, _, _ = integrator.apply(rhs, current_state, current_time, sample_time, dt, integrator_kwargs, *parameters)
        error = error + torch.linalg.norm(sample_state - current_state)/times.shape[0]

    if error.requires_grad:
        error.backward()
    return error
