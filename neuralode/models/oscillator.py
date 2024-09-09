import torch
import itertools


__all__ = ["OscillatorNet", "DrivenOscillatorNet"]


class OscillatorNet(torch.nn.Module):
    def __init__(self):
        # First we initialise the superclass, `torch.nn.Module`
        super().__init__()
        # Then we define the actual neural network
        # Most Neural Networks operate sequentially so they can be wrapped
        # inside a torch.nn.Sequential which takes each layer
        # as an argument.
        # Since we're only learning one matrix, we have
        # one layer, the `torch.nn.Linear`.
        # `torch.nn.Linear` stores a matrix and a bias which actually makes it
        # an Affine transformation rather than a purely linear transformation
        self.internal_net = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
        )

    def forward(self, x, t):
        # Our network only depends on x, but since it could also depend on t, we have
        # included it for completeness
        # Additionally, PyTorch layers and modules expect a batched tensor
        # ie. a tensor where the first dimension is over different samples
        # Since we don't depend on batches, we check if the input is 1-dimensional
        # And add a batch dimension as needed for the internal module
        if x.dim() == 1:
            return self.internal_net(x[None])[0]
        else:
            return self.internal_net(x)


# we define our network as a subclass of torch.nn.Module
# This allows PyTorch to appropriately track parameters
class DrivenOscillatorNet(torch.nn.Module):
    def __init__(self, number_of_hidden_neurons=32, number_of_hidden_layers=2,
                 intermediate_activation=torch.nn.Tanh, output_activation=torch.nn.Tanh):
        # First we initialise the superclass, `torch.nn.Module`
        super().__init__()
        # Then we define the actual neural network
        # Most Neural Networks operate sequentially so they can be wrapped
        # inside a torch.nn.Sequential which takes each layer
        # as an argument.
        # Since we're only learning one matrix, we have
        # one layer, the `torch.nn.Linear`.
        # `torch.nn.Linear` stores a matrix and a bias which actually makes it
        # an Affine transformation rather than a purely linear transformation
        hidden_layers = itertools.chain.from_iterable([
            (torch.nn.Linear(number_of_hidden_neurons, number_of_hidden_neurons), intermediate_activation())
            for _ in range(number_of_hidden_layers)])
        self.internal_net = torch.nn.Sequential(
            torch.nn.Linear(2, number_of_hidden_neurons),
            intermediate_activation(),
            *hidden_layers,
            torch.nn.Linear(number_of_hidden_neurons, 1),
            output_activation(),
        )

    def forward(self, x, t):
        if x.dim() == 1:
            return self.internal_net(x[None])[0]
        else:
            return self.internal_net(x)

