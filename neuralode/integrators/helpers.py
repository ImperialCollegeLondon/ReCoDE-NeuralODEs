import torch
import typing
import warnings

from neuralode import util


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
    if torch.any(torch.sign(t1 - t0) != torch.sign(dt)):
        # When dt points in a different direction to the direction of integration, this will cause issues in our integration loop
        # we either correct this or raise an error, we've elected to resolve this silently
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
    fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    time: torch.Tensor,
    step: torch.Tensor,
    tableau: torch.Tensor,
    intermediate_stages: torch.Tensor = None,
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
    for stage_index in range(intermediate_stages.shape[0]):
        c_coeff, a_coeff = tableau[stage_index][0], tableau[stage_index][1:]
        intermediate_stages[stage_index] = fn(
            # We use `compensated_sum` instead of `sum` to avoid truncation at each stage calculation
            state
            + step
            * util.compensated_sum(k * a for k, a in zip(intermediate_stages, a_coeff)),
            time + c_coeff * step,
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
