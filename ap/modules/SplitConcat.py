import torch
from torch import nn


class SplitConcat(nn.Module):

    def __init__(self, branches, concat_axis):

        super(SplitConcat, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.concat_axis = concat_axis

    def forward(self, x):

        assert len(self.branches) == len(x)

        y = []

        for idx, item in enumerate(x):
            y.append(self.branches[idx](item))

        return torch.cat(y, dim=self.concat_axis)
