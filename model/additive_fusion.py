"""Exact decomposition helpers for native single-output linear fusion layers.

These helpers expose evidence that is already present in a backbone.  They do
not add a trainable module or alter the inference graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import Conv2d


def decompose_single_output_pointwise_fusion(
    fusion_input: Tensor,
    layer: Conv2d,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(z, contributions, base)`` for a native 1x1 fusion conv.

    For an input ``x`` and a single-output pointwise convolution, the exact
    pre-sigmoid prediction is

    ``z = base + contributions.sum(dim=1, keepdim=True)``

    where ``contributions[:, i] = weight[i] * x[:, i]``.  All returned tensors
    remain connected to the original autograd graph.
    """

    if fusion_input.ndim != 4:
        raise ValueError("fusion_input must have shape [B,S,H,W]")
    if not isinstance(layer, Conv2d):
        raise TypeError("layer must be torch.nn.Conv2d")
    if layer.out_channels != 1:
        raise ValueError("fusion layer must have exactly one output channel")
    if layer.kernel_size != (1, 1) or layer.stride != (1, 1):
        raise ValueError("fusion layer must be a stride-1 pointwise convolution")
    if layer.padding != (0, 0) or layer.dilation != (1, 1) or layer.groups != 1:
        raise ValueError("fusion layer must be an ungrouped unpadded 1x1 convolution")
    if fusion_input.shape[1] != layer.in_channels:
        raise ValueError("fusion input channel count does not match the layer")

    weights = layer.weight.reshape(1, layer.in_channels, 1, 1)
    contributions = fusion_input * weights
    if layer.bias is None:
        base = fusion_input.new_zeros((1, 1, 1, 1))
    else:
        base = layer.bias.reshape(1, 1, 1, 1)
    z_full = base + contributions.sum(dim=1, keepdim=True)
    return z_full, contributions, base


class PointwiseFusionCapture:
    """Capture and decompose an existing pointwise fusion during a forward.

    Use this only around the native model forward.  Registering the hook has no
    parameters and removing it restores the model object exactly.
    """

    def __init__(self, layer: Conv2d) -> None:
        self.layer = layer
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._fusion_input: Tensor | None = None
        self._native_output: Tensor | None = None

    def _capture(self, _module: Conv2d, inputs: Sequence[Tensor]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("expected one fusion-layer input")
        self._fusion_input = inputs[0]

    def _capture_output(
        self, _module: Conv2d, _inputs: Sequence[Tensor], output: Tensor
    ) -> None:
        self._native_output = output

    def __enter__(self) -> "PointwiseFusionCapture":
        if self._handles:
            raise RuntimeError("capture is already active")
        self._fusion_input = None
        self._native_output = None
        self._handles = [
            self.layer.register_forward_pre_hook(self._capture),
            self.layer.register_forward_hook(self._capture_output),
        ]
        return self

    def decompose(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._fusion_input is None or self._native_output is None:
            raise RuntimeError("the fusion layer has not run inside this capture")
        _, contributions, base = decompose_single_output_pointwise_fusion(
            self._fusion_input, self.layer
        )
        # Use the layer's native result for decision tests.  Reconstructing the
        # same sum with elementwise operations can differ by a few FP32 ulps.
        return self._native_output, contributions, base

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        assert self._handles
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._fusion_input = None
        self._native_output = None
