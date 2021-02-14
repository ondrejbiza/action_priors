from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from .BasicBlock import BasicBlock
from .ResUBase import ResUBase
from .InHandConv import InHandConv


class ResUCat(nn.Module, ResUBase):
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

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
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

        place_q_values = self.place_q_values(feature_map_up_1)
        pick_q_values = self.pick_q_values(feature_map_up_1)
        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values
