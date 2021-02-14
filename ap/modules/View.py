from torch import nn


class View(nn.Module):

    def __init__(self, shape):

        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):

        return x.view((x.size()[0], *self.shape))
