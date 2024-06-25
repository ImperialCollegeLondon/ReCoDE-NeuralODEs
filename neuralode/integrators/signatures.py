import torch
import typing

integration_fn_signature = typing.Callable[
    [torch.Tensor, torch.Tensor, typing.Any], torch.Tensor
]
forward_method_signature = typing.Callable[
    [
        torch.autograd.function.FunctionCtx,
        integration_fn_signature,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

forward_method_nonadaptive_signature = typing.Callable[
    [
        torch.autograd.function.FunctionCtx,
        integration_fn_signature,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

backward_method_signature = typing.Callable[
    [
        torch.autograd.function.FunctionCtx,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[typing.Optional[torch.Tensor]],
]
