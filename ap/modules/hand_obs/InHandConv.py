from collections import OrderedDict
from torch import nn


class InHandConv(nn.Module):

    def __init__(self, patch_shape):
        super().__init__()
        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 64, kernel_size=3)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_pool2', nn.MaxPool2d(2)),
            ('cnn_conv3', nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

    def forward(self, in_hand):
        return self.in_hand_conv(in_hand)
