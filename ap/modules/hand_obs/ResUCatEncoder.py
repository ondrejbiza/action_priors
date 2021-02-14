import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from .ResUBase import ResUBase
from .BasicBlock import BasicBlock
from .InHandConv import InHandConv


class ResUCatEncoder(nn.Module, ResUBase):

    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100),
                 batch_norm=False):
        super().__init__()
        ResUBase.__init__(self, n_input_channel, batch_norm=batch_norm)
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

    def forward(self, states_and_actions, no_actions=False):

        # TODO: add an all actions option
        states = states_and_actions[0]
        actions = states_and_actions[1]

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

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))

        # I assume feature_map_up_1 is BxCxHxW
        x = feature_map_up_1

        # unpad output
        x = x[:, :, padding_width: -padding_width, padding_width: -padding_width]

        if no_actions:
            return x

        # select only the embedding for the selected action
        x = x.reshape((x.size(0), x.size(1), x.size(2) * x.size(3)))
        x = x[list(range(x.size(0))), :, actions]

        return x
