from collections import OrderedDict
from torch import nn
from .BasicBlock import BasicBlock


class ResUBaseDown:
    def __init__(self, n_input_channel=1):
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'enc-conv5',
                        nn.Conv2d(512, 256, kernel_size=1, bias=False)
                    )
                ]
            )
        )

