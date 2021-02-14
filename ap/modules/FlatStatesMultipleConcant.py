import torch
from torch import nn


class FlatStatesMultipleConcat(nn.Module):

    def __init__(self):

        super(FlatStatesMultipleConcat, self).__init__()

    def forward(self, x):

        flats = []

        for xx in x:
            flats.append(self.flatten_states_(xx))

        return self.concat_(flats)

    def flatten_states_(self, states):

        return torch.flatten(states, 1)

    def concat_(self, l):

        return torch.cat(l, dim=1)
