import torch
import typing


def partial_compensated_sum(
    next_value: torch.Tensor,
    partial_sum_truncated_bits: tuple[torch.Tensor, torch.Tensor] = None,
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
        partial_sum = torch.zeros_like(next_value)
        truncated_bits = torch.zeros_like(next_value)
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


def compensated_sum(iterable_to_sum: typing.Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Functional equivalent to the python function `sum` but
    uses the Kahan-Babuška-Neumaier scheme to track truncated bits.

    Usage is the same as `sum`, but only works with pytorch tensors.

    References:
        [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    :param iterable_to_sum: any kind of iterable including generators that returns tensors
    :return: the compensated sum
    """
    partial_sum, truncated_bits = None, None
    for v in iterable_to_sum:
        if partial_sum is None:
            partial_sum, truncated_bits = torch.zeros_like(v), torch.zeros_like(v)
        partial_sum, truncated_bits = partial_compensated_sum(
            v, (partial_sum, truncated_bits)
        )
    return partial_sum + truncated_bits  # Add truncated bits back to the sum
