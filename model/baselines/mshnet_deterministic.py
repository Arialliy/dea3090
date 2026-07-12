"""Parameter-identical MSHNet variant with deterministic max-reduction backward."""

import torch

from .mshnet_official import (
    ChannelAttention as OfficialChannelAttention,
    MSHNet as OfficialMSHNet,
    ResNet as OfficialResNet,
)


class ChannelAttention(OfficialChannelAttention):
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        spatial_max = torch.amax(x, dim=(-2, -1), keepdim=True)
        max_out = self.fc2(self.relu1(self.fc1(spatial_max)))
        return self.sigmoid(avg_out + max_out)


class ResNet(OfficialResNet):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, stride)
        self.ca = ChannelAttention(out_channels)


class MSHNet(OfficialMSHNet):
    def __init__(self, input_channels, block=ResNet):
        super().__init__(input_channels, block=block)


__all__ = ["ChannelAttention", "ResNet", "MSHNet"]
