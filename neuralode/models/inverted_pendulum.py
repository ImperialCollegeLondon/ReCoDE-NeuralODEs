import torch
import itertools

__all__ = ["CartPoleNet"]


class CartPoleNet(torch.nn.Module):
    def __init__(self, max_force=256.0, num_hidden_neurons=128, num_hidden_layers=4):
        super().__init__()
        hidden_layers = itertools.chain.from_iterable(
            [
                (
                    torch.nn.Linear(num_hidden_neurons, num_hidden_neurons),
                    torch.nn.Tanh(),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.internal_net = torch.nn.Sequential(
            torch.nn.Linear(8, num_hidden_neurons),
            torch.nn.Tanh(),
            *hidden_layers,
            torch.nn.Linear(num_hidden_neurons, 1, bias=False),
        )
        self.max_force = torch.nn.Parameter(
            torch.tensor(max_force), requires_grad=False
        )

    def forward(self, x, t):
        sin_angle, cos_angle = torch.sin(x[..., 0]), torch.cos(x[..., 0])
        encoded_x = torch.stack(
            [
                x[..., 0],
                sin_angle,
                cos_angle,
                x[..., 1],
                -x[..., 1] * cos_angle,
                x[..., 1] * sin_angle,
                x[..., 2],
                x[..., 3],
            ],
            dim=-1,
        )
        if x.dim() == 1:
            res = self.internal_net(encoded_x[None])[0]
        else:
            res = self.internal_net(encoded_x)
        force = torch.sin(0.5 * torch.pi * res)
        return force * self.max_force
