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


def get_simple_harmonic_oscillator_matrix(
    frequency: torch.Tensor, damping: torch.Tensor
) -> torch.Tensor:
    """
    Computes the matrix of a simple harmonic oscillator.

    :param frequency: The frequency of the oscillator.
    :param damping: The damping coefficient of the oscillator.
    :return:
    """
    return torch.stack(
        [
            # no x term for the derivative of x as it is equal to v
            torch.stack(
                [torch.zeros_like(frequency), torch.ones_like(frequency)], dim=-1
            ),
            # first we have the omega^2 term, then the 2*zeta*omega term
            torch.stack([-(frequency**2), -2 * frequency * damping], dim=-1),
        ],
        dim=-2,
    )


def simple_harmonic_oscillator(
    x: torch.Tensor, t: torch.Tensor, frequency: torch.Tensor, damping: torch.Tensor
):
    """
    Computes the derivative vector of a simple harmonic oscillator with given frequency and damping.

    :param x: The current state.
    :param t: The current time (unused).
    :param frequency: The frequency of the oscillator.
    :param damping: The damping coefficient of the oscillator.
    :return: The time derivative of the SHA.
    """
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
    return einops.einsum(A, x, "... row col,... col->... row")


def inverted_pendulum(state, _, force, cart_mass, pole_mass, pole_length, gravity, friction_coefficient_of_cart,
                      friction_coefficient_of_pole):
    """
    Implementation of the cart-pole system described in: https://coneural.org/florian/papers/05_cart_pole.pdf

    Has fast-path whenever cart friction and pole friction are both zero.

    :param state: The state of the cart-pole system.
    :param _:
    :param force: The force applied to the cart in N, with (+)ive being rightward and (-)ive being leftward.
    :param cart_mass: The mass of the cart in kg
    :param pole_mass: The mass of the pole in kg
    :param pole_length: The length of the pole in m
    :param gravity: The gravitational acceleration in m/s^2
    :param friction_coefficient_of_cart: The friction coefficient of the cart.
    :param friction_coefficient_of_pole: The friction coefficient of the pole.
    :return:
    """
    theta, theta_dot, _, x_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]

    dtheta = theta_dot
    dx = x_dot

    stheta, ctheta = theta.sin(), theta.cos()

    theta_dot_sq = theta_dot.square()
    total_mass = cart_mass + pole_mass * stheta.square()

    if torch.all(friction_coefficient_of_cart == 0.0) and torch.all(friction_coefficient_of_pole == 0.0):
        pole_moment_of_inertia = pole_mass * pole_length / 2
        counter_force = (force + pole_moment_of_inertia * stheta * theta_dot_sq) / total_mass
        dtheta_dot = stheta * gravity - ctheta * counter_force
        dtheta_dot = dtheta_dot * 2 / pole_length / (4.0 / 3.0 - pole_mass * ctheta.square() / total_mass)
        dx_dot = counter_force - pole_moment_of_inertia * ctheta * dtheta_dot / total_mass
    else:
        sgn_xdot = torch.where(x_dot.abs() < 1e-7, torch.zeros_like(x_dot), torch.sign(x_dot))
        paren1_pre = -force - pole_mass * pole_length / 2 * theta_dot_sq * (
                    stheta + friction_coefficient_of_cart * ctheta * sgn_xdot) + friction_coefficient_of_cart * gravity * sgn_xdot

        dtheta_dot_common = gravity * stheta - (friction_coefficient_of_pole / pole_mass) * x_dot * 2 / pole_length
        dtheta_dot = dtheta_dot_common + ctheta * paren1_pre
        dtheta_dot = dtheta_dot * 2 / pole_length / (
                    4.0 / 3.0 - pole_mass * ctheta / total_mass * (ctheta - friction_coefficient_of_cart * sgn_xdot))

        cart_normal_force = total_mass * gravity - pole_mass * pole_length / 2 * (dtheta_dot * stheta + theta_dot_sq * ctheta)

        corr_needed = (torch.sign(cart_normal_force) < 0) & (friction_coefficient_of_cart > 0.0)
        if torch.any(corr_needed):
            sgn_xdot = torch.sign(cart_normal_force * x_dot)
            paren1_post = -force - pole_mass * pole_length / 2 * theta_dot_sq * (
                        stheta + friction_coefficient_of_cart * ctheta * sgn_xdot) + friction_coefficient_of_cart * gravity * sgn_xdot

            dtheta_dot_pre = dtheta_dot_common + ctheta * paren1_post
            dtheta_dot = torch.where(corr_needed, dtheta_dot_pre * 2 / pole_length / (
                        4.0 / 3.0 - pole_mass * ctheta / total_mass * (
                            ctheta - friction_coefficient_of_cart * sgn_xdot)), dtheta_dot)

            cart_normal_force = torch.where(corr_needed, (cart_mass + pole_mass) * gravity - pole_mass * pole_length / 2 * (
                        dtheta_dot * stheta + theta_dot_sq * ctheta), cart_normal_force)

        dx_dot = (force + pole_mass * (
                    theta_dot_sq * stheta - dtheta_dot * ctheta) - friction_coefficient_of_cart * cart_normal_force * sgn_xdot) / (
                             cart_mass + pole_mass)

    return torch.stack([
        dtheta,
        dtheta_dot,
        dx,
        dx_dot
    ], dim=-1)
