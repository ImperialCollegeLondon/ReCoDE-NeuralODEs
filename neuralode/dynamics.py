import torch
import einops


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


def get_simple_harmonic_oscillator_matrix(frequency, damping):
    return torch.stack([
        # no x term for the derivative of x as it is equal to v
        torch.stack([torch.zeros_like(frequency), torch.ones_like(frequency)], dim = -1),
        # first we have the omega^2 term, then the 2*zeta*omega term
        torch.stack([-frequency ** 2, -2 * frequency * damping], dim = -1),
        ], dim = -2)


def simple_harmonic_oscillator(x, t, frequency, damping):
    # The dynamics above can easily be represented as a matrix multiplication
    # First the matrix with the corresponding terms
    A = get_simple_harmonic_oscillator_matrix(frequency, damping)
    # We implement the matrix multiplication using einops
    # This is not necessarily the most efficient, but it allows
    # us to track the exact operation without worrying about the shapes
    # of our tensors too much
    # You can read '...,ij,...j->...i' as:
    #   - The first argument is a tensor with arbitrary dimensions, but
    #   the last two of which are of interest, labelled as 'i' and 'j'
    #   - The second argument is a tensor with arbitrary dimensions, but
    #   the last of which is commensurate with the number of rows of the input matrix
    #   - Take the sum of A[...,i,j]*x[...,j] over all 'j' and the output will be indexed
    #   by 'i' in the last dimension
    return einops.einsum(A, x, '... row col,... col->... row')
