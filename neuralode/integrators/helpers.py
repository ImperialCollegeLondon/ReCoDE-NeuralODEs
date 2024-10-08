import torch
import typing
import warnings
import math

from neuralode import util
from neuralode.integrators import signatures


def ensure_tolerance(tol, x, tol_shortname="tol", tol_name="Tolerance"):
    """Ensures that the tolerances are valid and compatible with the input state"""
    epsilon = torch.finfo(x.dtype).eps
    sqrt_epsilon = torch.finfo(x.dtype).eps ** 0.5
    tol = torch.abs(
        tol
    )  # Need to ensure that the tolerance is a strictly positive number
    if tol.dim() != 0:
        # We can set per-component tolerances, but that means that the shapes of x0 and atol/rtol are compatible.
        # To test this, we try to add the tolerance to the initial state and check for errors
        try:
            _ = x.clone() + tol.clone()
        except RuntimeError:
            raise ValueError(
                f"Expected shape of x0 and {tol_shortname} to be compatible, got {x.shape} and {tol.shape} respectively"
            )
    if torch.any((tol < epsilon) & (tol != 0.0)):
        # If any component of `tol` is too small, we adjust the values to be compatible with the datatype
        # To preserve differentiability, we divide out the detached value of `tol` and multiply by the epsilon
        warnings.warn(
            f"Absolute tolerance is too small for tensor dtype: {x.dtype}. Set {tol_shortname}={epsilon}.",
            RuntimeWarning,
        )
        tol = torch.where(
            (tol < epsilon) & (tol != 0.0), (tol / tol.detach()) * epsilon, tol
        )
    elif torch.any(tol < sqrt_epsilon):
        # When the tolerances are tiny (roughly eps^0.5), truncation error can start to dominate
        # while we are compensating for this using Kahan summation, the user should be informed that this may cause issues.
        warnings.warn(
            f"{tol_name.title()} is smaller than the square root of the epsilon for {x.dtype}, this may increase truncation error",
            RuntimeWarning,
        )
    return tol


def ensure_timestep(dt, t0, t1):
    """Ensures that the timestep is valid given the initial and final times"""
    if torch.any(torch.abs(t1 - t0) < torch.abs(dt)):
        # If the timestep is too large, we clamp it to the integration interval
        dt = t1 - t0

    if torch.any(torch.sign(t1 - t0) != torch.sign(dt)):
        # When dt points in a different direction to the direction of integration, this will cause issues in our
        # integration loop we either correct this or raise an error, we've elected to resolve this silently
        warnings.warn(
            f"Different sign of (t1 - t0) and dt: {t1 - t0} and {dt}, correcting...",
            UserWarning,
        )
        dt = torch.copysign(dt, t1 - t0)

    if torch.any((t0 + dt) == t0):
        # If the timestep is too small to be measurable, then we need to adjust it accordingly
        dt = util.next_value(t0, dt) - t0

    return dt


def compute_step(
    fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    time: torch.Tensor,
    step: torch.Tensor,
    tableau: torch.Tensor,
    intermediate_stages: torch.Tensor,
    is_adaptive: bool = False,
):
    """Computes a single integration step given the integration scheme specified by `tableau`

    :param fn: Callable for the right-hand side of the numerical integration
    :param state: Current state of the system
    :param time: Current time of the system
    :param step: Step size
    :param tableau: Butcher Tableau specifying the integration routine
    :param intermediate_stages:
    :param is_adaptive:
    :return: A tuple containing the change in the state and the step size taken
    """
    # Since we've seen that compensated summation does not significantly improve the results
    # for floating-point values with less than 64 bits of precision,
    # we can switch to using the faster `torch.sum` function
    sum_func = (
        util.compensated_sum
        if torch.finfo(state.dtype).bits > 32
        else lambda x: torch.sum(x, dim=0)
    )
    intermediate_stages = intermediate_stages.detach()

    for stage_index in range(intermediate_stages.shape[0]):
        c_coeff, a_coeff = (
            tableau[stage_index][0],
            tableau[stage_index][1:stage_index+1],
        )
        ks = state
        if stage_index > 0:
            ks = ks + step * sum_func((a_coeff * intermediate_stages[:stage_index].transpose(0, -1)).transpose(0, -1))
        intermediate_stages[stage_index] = fn(
            # We use `compensated_sum` instead of `sum` to avoid truncation at each stage calculation
            ks,
            time + c_coeff * step,
        )
    lower_order_estimate = step * sum_func(
        (tableau[-1, 1:] * intermediate_stages.transpose(0, -1)).transpose(0, -1)
    )
    # To have a valid value, we set `higher_order_estimate` to the same value as `lower_order_estimate`
    # Further down, this will simplify the code as we won't have to account for invalid values
    higher_order_estimate = lower_order_estimate
    if is_adaptive:
        higher_order_estimate = step * sum_func(
            (tableau[-2, 1:] * intermediate_stages.transpose(0, -1)).transpose(0, -1)
        )
    # From a numerical perspective, this implementation is not necessarily ideal as
    # we can lose precision when subtracting the two solutions. A more numerically accurate
    # implementation would have one row `b_i` coefficients and another row the coefficients
    # for computing the error directly
    return lower_order_estimate, higher_order_estimate, step


def construct_adjoint_fn(
        forward_fn: signatures.integration_fn_signature,
        state_shape: tuple[int] | torch.Size,
        gradient_clipping: torch.Tensor | None = None
) -> signatures.integration_fn_signature:
    state_size = math.prod(state_shape)

    def adjoint_fn(
            packed_state: torch.Tensor,
            adj_time: torch.Tensor,
            *dynamic_args: tuple[torch.Tensor],
    ) -> torch.Tensor:
        # Unpack the state variables
        y = packed_state[:state_size].reshape(state_shape)
        adj_lagrange = packed_state[state_size:2 * state_size].reshape(state_shape)

        # We want the product of the jacobian of dy/dt wrt. each of the parameters multiplied by the adjoint variables
        # This is the Jacobian-vector product which can be directly computed through PyTorch autograd
        # using the `torch.autograd.grad` function and passing the adjoint variables as the incoming gradients
        dy, final_grads = torch.autograd.functional.vjp(forward_fn, (y, adj_time, *dynamic_args), adj_lagrange,
                                                        create_graph=packed_state.requires_grad)
        adj_lagrange_derivatives = final_grads[0].ravel()[:state_size]
        parameter_derivatives_of_fn = final_grads[2:]

        d_adj = torch.cat(
            [
                dy.ravel(),
                -adj_lagrange_derivatives,
            ],
            dim=0,
        )
        if len(parameter_derivatives_of_fn) > 0:
            parameter_derivatives_of_fn = torch.cat(
                [i.ravel() for i in parameter_derivatives_of_fn]
            )
            if gradient_clipping is not None:
                parameter_derivatives_of_fn = torch.where(
                    torch.isfinite(parameter_derivatives_of_fn),
                    parameter_derivatives_of_fn,
                    0.0
                )
                parameter_derivatives_of_fn = parameter_derivatives_of_fn.clamp(min=-gradient_clipping.abs(),
                                                                                max=gradient_clipping.abs())
            d_adj = torch.cat([d_adj, -parameter_derivatives_of_fn])
        return d_adj

    return adjoint_fn


def adapt_adaptive_timestep(
    current_dt: torch.Tensor,
    current_error: torch.Tensor,
    max_error: torch.Tensor,
    method_order: int,
    min_timestep: torch.Tensor | None=None,
    max_timestep: torch.Tensor | None=None
):
    relative_error = torch.where(max_error != 0.0, current_error / max_error, 0.0)
    relative_error = torch.where((current_error != 0.0) & (max_error == 0.0), 1.0, relative_error).max()
    step_correction = torch.where(
        relative_error != 0.0,
        0.8 * (1.0/relative_error) ** (1 / method_order),
        1.05,
    )
    step_correction = torch.clamp(step_correction, min=1e-4, max=1.5)
    # Based on the error, we correct the step size
    new_dt = torch.clamp(step_correction * current_dt, min=min_timestep, max=max_timestep)
    return new_dt, relative_error >= 1.0
