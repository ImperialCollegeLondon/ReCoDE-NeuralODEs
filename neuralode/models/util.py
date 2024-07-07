import torch

__all__ = ["init_weights"]


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.5)
        if m.bias is not None:
            m.bias.data.normal_(0.0, 0.01)
