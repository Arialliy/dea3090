"""Function-preserving input-domain lifts for canonical MSHNet ResNet blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.baselines.mshnet_deterministic import ResNet


def widen_resnet_input_with_zeros(
    source: ResNet,
    new_input_channels: int,
) -> ResNet:
    """Append zero-kernel coordinates while preserving the native block.

    Both the main 3x3 convolution and the projection shortcut are widened.
    The experiment RNG stream is restored because every newly drawn parameter
    is overwritten by the exact function-preserving embedding.
    """

    input_channels = source.conv1.in_channels
    output_channels = source.conv1.out_channels
    new_input_channels = int(new_input_channels)
    if new_input_channels <= input_channels:
        raise ValueError("new_input_channels must exceed the native input width")
    stride = int(source.conv1.stride[0])
    rng_state = torch.get_rng_state()
    try:
        target = ResNet(new_input_channels, output_channels, stride=stride)
        if source.shortcut is not None and target.shortcut is None:
            target.shortcut = nn.Sequential(
                nn.Conv2d(
                    new_input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(output_channels),
            )
    finally:
        torch.set_rng_state(rng_state)
    if (source.shortcut is None) != (target.shortcut is None):
        raise ValueError("widening changed the native shortcut topology")

    source_state = source.state_dict()
    target_state = target.state_dict()
    widened_keys = {"conv1.weight", "shortcut.0.weight"}
    for key, target_value in target_state.items():
        if key in widened_keys:
            if key not in source_state:
                raise ValueError(f"source block has no required tensor {key}")
            source_value = source_state[key]
            if target_value.shape[1] != new_input_channels:
                raise ValueError(f"unexpected widened shape for {key}")
            embedded = torch.zeros_like(target_value)
            embedded[:, : source_value.shape[1]].copy_(source_value)
            target_state[key] = embedded
        else:
            if key not in source_state or target_value.shape != source_state[key].shape:
                raise ValueError(f"cannot embed native block tensor {key}")
            target_state[key] = source_state[key].clone()
    target.load_state_dict(target_state, strict=True)
    return target


__all__ = ["widen_resnet_input_with_zeros"]
