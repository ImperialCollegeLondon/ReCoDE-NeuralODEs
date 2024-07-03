import torch


def exponential_fn(x, t, decay_constant=-1.0):
    """
    The derivative of an exponential system
    :param x: the current state of the system
    :param t: the current time of the system
    :param decay_constant: the decay constant of the system
    :return: the derivative of the exponential system at (x,t)
    """
    return decay_constant * x


def exponential_fn_solution(initial_state, t, decay_constant=-1.0):
    """
    The solution of an exponentially decaying system
    :param initial_state: the initial state of the system
    :param t: the time at which the solution is desired
    :param decay_constant: the decay constant of the system
    :return: the state at time t
    """
    return initial_state * torch.exp(decay_constant * t)
