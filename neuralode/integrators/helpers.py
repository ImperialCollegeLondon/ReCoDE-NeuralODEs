import math
import torch
import warnings

from neuralode import util
from neuralode.integrators import signatures


def ensure_tolerance(tol: torch.Tensor, x0: torch.Tensor):
    tol = torch.abs(tol)
    if tol.dim() != 0:
        # We can set per-component tolerances, but that means that the shapes of `x0` and `tol` must be compatible.
        # To test this, we try to add `tol` to `x0`
        try:
            _ = x0.clone() + tol.clone()
        except RuntimeError:
            raise ValueError(
                f"Expected shape of x0 and tol to be compatible, got {x0.shape} and {tol.shape} respectively"
                )
    
    if torch.any((tol < torch.finfo(x0.dtype).eps) & (tol != 0.0)):
        # If any component of `tol` is too small, we adjust the values to be compatible with the datatype
        # To preserve differentiability, we divide out the detached value of `tol` and multiply by the epsilon
        warnings.warn(
            f"Tolerance is too small for tensor dtype: {x0.dtype}. Set tol={torch.finfo(x0.dtype).eps}.",
            RuntimeWarning,
            )
        tol = torch.where(
            (tol < torch.finfo(x0.dtype).eps) & (tol != 0.0),
            (tol / tol.detach()) * torch.finfo(x0.dtype).eps,
            tol,
            )
    elif torch.any(tol < torch.finfo(x0.dtype).eps ** 0.5):
        # When the tolerances are tiny (roughly eps^0.5), truncation error can start to dominate while we are
        # compensating for this using Kahan summation, the user should be informed that this may cause issues.
        warnings.warn(
            f"Tolerance is smaller than the square root of the epsilon for {x0.dtype}, this may increase "
            f"truncation error",
            RuntimeWarning,
            )
    return tol


def ensure_timestep(t0: torch.Tensor, t1: torch.Tensor, dt: torch.Tensor):
    if torch.any(torch.abs(t1 - t0) < torch.abs(dt)):
        # If the timestep is too large, we clamp it to the integration interval
        dt = t1 - t0
    
    if torch.any(torch.sign(t1 - t0) != torch.sign(dt)):
        # When dt points in a different direction to the direction of integration, this will cause issues in our
        # integration loop we either correct this or raise an error, we've elected to resolve this silently
        warnings.warn(
            f"Different sign of (t1 - t0) and dt: {t1 - t0} and {dt}, correcting...",
            RuntimeWarning,
            )
        dt = torch.copysign(dt, t1 - t0)
    
    if torch.any((t0 + dt) == t0):
        # If the timestep is too small to be measurable, then we need to adjust it accordingly
        dt = util.next_value(t0, dt) - t0
    
    return dt


def compute_step(
        fn,
        state,
        time,
        step,
        tableau: torch.Tensor,
        /,
        *,
        additional_dynamic_args=tuple(),
        intermediate_stages: torch.Tensor = None,
        is_adaptive: bool = False,
        ):
    for stage_index in range(intermediate_stages.shape[0]):
        c_coeff, a_coeff = tableau[stage_index][0], tableau[stage_index][1:]
        intermediate_stages[stage_index] = fn(
            # We use `compensated_sum` instead of `sum` to avoid truncation at each stage calculation
            state
            + step
            * util.compensated_sum(k * a for k, a in zip(intermediate_stages, a_coeff)),
            time + c_coeff * step,
            *additional_dynamic_args,
            )
    lower_order_estimate = step * util.compensated_sum(
        k * b for k, b in zip(intermediate_stages, tableau[-1, 1:])
        )
    # To have a valid value, we set `higher_order_estimate` to the same value as `lower_order_estimate`
    # Further down, this will simplify the code as we won't have to account for invalid values
    higher_order_estimate = lower_order_estimate
    if is_adaptive:
        higher_order_estimate = step * util.compensated_sum(
            k * b for k, b in zip(intermediate_stages, tableau[-2, 1:])
            )
    # From a numerical perspective, this implementation is not necessarily ideal as
    # we can lose precision when subtracting the two solutions. A more numerically accurate
    # implementation would have one row `b_i` coefficients and another row the coefficients
    # for computing the error directly
    return lower_order_estimate, higher_order_estimate, step


def construct_adjoint_fn(forward_fn: signatures.integration_fn_signature,
                         state_shape: tuple[int] | torch.Size) -> signatures.integration_fn_signature:
    state_size = math.prod(state_shape)
    
    def adjoint_fn(packed_state: torch.Tensor, adj_time: torch.Tensor,
                   *dynamic_args: tuple[torch.Tensor]) -> torch.Tensor:
        # Unpack the state variables
        with torch.set_grad_enabled(True):
            packed_state = packed_state.requires_grad_(True)
            adj_lagrange = packed_state[state_size:2 * state_size].reshape(state_shape)
            dy = forward_fn(packed_state[:state_size].reshape(state_shape), adj_time, *dynamic_args).ravel()
        # We want the product of the jacobian of dy/dt wrt. each of the parameters multiplied by the adjoint variables
        # This is the Jacobian-vector product which can be directly computed through PyTorch autograd
        # using the `torch.autograd.grad` function and passing the adjoint variables as the incoming gradients
        
        final_grads = util.masked_grad(dy, [packed_state] + list(dynamic_args), adj_lagrange,
                                       create_graph = True, materialize_grads = True, allow_unused = True)
        adj_lagrange_derivatives = final_grads[0][:state_size]
        parameter_derivatives_of_fn = final_grads[1:]
        
        d_adj = torch.cat([
            dy,
            -adj_lagrange_derivatives,
            ], dim = 0)
        if len(parameter_derivatives_of_fn) > 0:
            parameter_derivatives_of_fn = torch.cat([i.ravel() for i in parameter_derivatives_of_fn])
            d_adj = torch.cat([
                d_adj,
                -parameter_derivatives_of_fn
                ])
        return d_adj
    
    return adjoint_fn

