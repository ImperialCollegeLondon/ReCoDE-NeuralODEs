import torch
import itertools

__all__ = ["CartPoleNet"]


class CartPoleNet(torch.nn.Module):
    def __init__(self, max_force=160.0, num_hidden_neurons=128, num_hidden_layers=4):
        super().__init__()
        hidden_layers = itertools.chain.from_iterable(
            [
                (
                    torch.nn.Linear(num_hidden_neurons, num_hidden_neurons),
                    torch.nn.CELU(),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.internal_net = torch.nn.Sequential(
            torch.nn.Linear(4, num_hidden_neurons),
            torch.nn.CELU(),
            *hidden_layers,
            torch.nn.Linear(num_hidden_neurons, 1, bias=False),
        )
        self.max_force = torch.nn.Parameter(
            torch.tensor(max_force), requires_grad=False
        )

    def forward(self, x, t):
        encoded_x = torch.stack(
            [
                x[..., 0],
                x[..., 1],
                x[..., 2],
                x[..., 3],
            ],
            dim=-1,
        )
        if x.dim() == 1:
            force = self.internal_net(encoded_x[None])[0]
        else:
            force = self.internal_net(encoded_x)
        force = force + (force.clamp(min=-1.0, max=1.0) - force).detach()
        return force * self.max_force
