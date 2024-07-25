import torch
import typing

integration_fn_signature = typing.Callable[
    [torch.Tensor, torch.Tensor, *list[typing.Any]], torch.Tensor
]

setup_context_signature = typing.Callable[
    [torch.autograd.function.FunctionCtx, typing.Any, typing.Any], None
]

forward_method_signature = typing.Callable[
    [
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
