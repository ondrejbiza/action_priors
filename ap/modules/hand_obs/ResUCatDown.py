import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from .ResUBaseDown import ResUBaseDown
from .BasicBlock import BasicBlock
from .InHandConv import InHandConv


class ResUCatDown(nn.Module, ResUBaseDown):

    def __init__(self, n_input_channel=1, patch_shape=(1, 24, 24)):
        super().__init__()
        ResUBaseDown.__init__(self, n_input_channel)
        self.conv_cat_in_hand = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-res6',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

        self.in_hand_conv = InHandConv(patch_shape)
        self.output_size = 32

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, states):

        obs = states[0]
        in_hand = states[1]

        # pad obs
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)

        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, in_hand_out), dim=1))

        return feature_map_16
