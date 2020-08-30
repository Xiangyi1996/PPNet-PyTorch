# -*- coding: utf-8 -*-
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import torch.nn as nn
import torch


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, atrous_rates):
        super(_ASPP, self).__init__()

        for i, rate in enumerate(atrous_rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


