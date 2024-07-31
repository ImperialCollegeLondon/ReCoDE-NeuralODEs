import torch
import typing


def partial_compensated_sum(
    next_value: torch.Tensor,
    partial_sum_truncated_bits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    An iteratively callable implementation of the Kahan-Babuška-Neumaier scheme to
    track truncated bits in a sum.

    Usage::
        >>> values_to_sum = [1e38,1,-1e38]
        >>> partial_sum, truncated_bits = partial_compensated_sum(values_to_sum[0])
        >>> partial_sum, truncated_bits = partial_compensated_sum(values_to_sum[1], (partial_sum, truncated_bits))
        >>> partial_sum, truncated_bits = partial_compensated_sum(values_to_sum[2], (partial_sum, truncated_bits))
        >>> final_sum = partial_sum + truncated_bits
        >>> print(final_sum)
            1

    References:
        [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    :param next_value: The next value to add to the partial sum
    :param partial_sum_truncated_bits: A tuple of (partial_sum, truncated_bits)
    :return: the updated partial sum and truncated bits
    """
    if partial_sum_truncated_bits is None:
        partial_sum, truncated_bits = next_value, 0.0
    else:
        partial_sum, truncated_bits = partial_sum_truncated_bits
        temporary_partial_sum = partial_sum + next_value
        truncated_bits = truncated_bits + torch.where(
            torch.abs(partial_sum) >= torch.abs(next_value),
            # When the magnitude of the partial sum is larger, truncation occurs for v and vice versa when v is larger
            (partial_sum - temporary_partial_sum) + next_value,
            # First the negation of the truncated value of v is computed from the partial sum and
            # the temporary partial sum, and then adding it to v gives the truncated bits
            (next_value - temporary_partial_sum)
            + partial_sum,  # As before, but the role of v and partial_sum are swapped
        )
        partial_sum = temporary_partial_sum
    return partial_sum, truncated_bits


def compensated_sum(iterable_to_sum: typing.Iterable[torch.Tensor]) -> torch.Tensor|float:
    """
    Functional equivalent to the python function `sum` but
    uses the Kahan-Babuška-Neumaier scheme to track truncated bits.

    Usage is the same as `sum`, but only works with pytorch tensors.

    References:
        [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    :param iterable_to_sum: any kind of iterable including generators that returns tensors
    :return: the compensated sum
    """
    partial_sum_truncated_bits = None
    for v in iterable_to_sum:
        partial_sum_truncated_bits = partial_compensated_sum(
            v, partial_sum_truncated_bits
        )
    if partial_sum_truncated_bits is not None:
        return partial_sum_truncated_bits[0] + partial_sum_truncated_bits[1]  # Add truncated bits back to the sum
    else:
        return 0.0


def next_value(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Returns the next smallest number in the direction of `direction`.

    If `direction` is positive, then it is the smallest number larger than `v`.
    If `direction` is negative, then it is the larger number smaller than `v`.

    :param v:
    :param direction:
    :return: The next value
    """
    nearest_base_2_val = v.abs().log2().floor().exp2()
    return v + torch.copysign(
        nearest_base_2_val * torch.finfo(v.dtype).eps + torch.finfo(v.dtype).eps,
        direction,
    )


def masked_grad(
    outputs: torch.Tensor, grad_vars: list[torch.Tensor], *args, **kwargs
) -> list[torch.Tensor | None]:
    """
    Computes a masked version of `torch.autograd.grad` reducing the issues with passing in variables
    that don't require grad and automagically outputting `None` for variables that don't have a gradient associated
    with them.

    :param outputs: The values whose gradients are needed
    :param grad_vars: The variables with respect to which the gradients should be computed
    :param args: positional arguments for `torch.autograd.grad`
    :param kwargs: keyword arguments for `torch.autograd.grad`
    :return: list of gradients of the outputs wrt. every input in `grad_vars`
    """
    grad_vars_mask = [i.requires_grad for i in grad_vars]

    final_grads: list[torch.Tensor | None] = [None for i in grad_vars]

    if any(grad_vars_mask):
        grad_vars_needed = [
            grad_vars[idx]
            for idx, grad_needed in enumerate(grad_vars_mask)
            if grad_needed
        ]
        grad_results = torch.autograd.grad(
            outputs, grad_vars_needed, *args, **kwargs
        )
        if grad_results:
            for idx, (grad_needed, grad_v) in enumerate(zip(grad_vars_mask, grad_vars)):
                if grad_needed:
                    final_grads[idx] = grad_results[0]
                    grad_results = grad_results[1:]
                else:
                    final_grads[idx] = (
                        torch.zeros_like(grad_v)
                        if kwargs.get("materialize_grads", False)
                        else None
                    )
    return final_grads


def interp_hermite(
    xs: torch.Tensor, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor
) -> torch.Tensor:
    """
    Piecewise Hermite cubic spline interpolation.

    Based on the implementation shown in [1].

    [1]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline

    :param xs: Points to interpolate to
    :param x: Nodes of the input values
    :param y: The function value at each node
    :param dy: The function slope at each node
    :return: The function interpolated to the points `xs
    """

    sample_x = xs.ravel()
    xidx = torch.searchsorted(x, sample_x)
    xidx = torch.clamp(xidx, 0, x.shape[0] - 2)

    x0 = x[xidx]
    x1 = x[xidx + 1]

    y0 = y[xidx].transpose(0, -1)
    y1 = y[xidx + 1].transpose(0, -1)
    dy0 = dy[xidx].transpose(0, -1)
    dy1 = dy[xidx + 1].transpose(0, -1)

    dx = x1 - x0
    t = (sample_x - x0) / dx
    t2 = t.square()
    t3 = t * t.square()

    h00 = (2 * t3 - 3 * t2 + 1)
    h10 = ((t3 - 2 * t2 + t) * dx)
    h01 = (-2 * t3 + 3 * t2)
    h11 = ((t3 - t2) * dx)

    return (h00 * y0 + h10 * dy0 + h01 * y1 + h11 * dy1).transpose(0, -1)
