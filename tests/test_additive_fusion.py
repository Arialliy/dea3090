from __future__ import annotations

import torch
import torch.nn as nn

from model.additive_fusion import (
    PointwiseFusionCapture,
    decompose_single_output_pointwise_fusion,
)
from model.counterfactual_responsibility import (
    counterfactual_responsibility_suppression,
)


def test_pointwise_decomposition_matches_native_forward_and_gradients() -> None:
    torch.manual_seed(7)
    layer = nn.Conv2d(6, 1, kernel_size=1, bias=True)
    native_input = torch.randn(2, 6, 5, 4, requires_grad=True)
    decomposed_input = native_input.detach().clone().requires_grad_(True)

    native = layer(native_input)
    z_full, contributions, base = decompose_single_output_pointwise_fusion(
        decomposed_input, layer
    )

    # The native convolution and explicit reduction can differ only by FP32
    # accumulation order; selection code uses the captured native result.
    torch.testing.assert_close(z_full, native, rtol=0, atol=2e-7)
    torch.testing.assert_close(
        z_full, base + contributions.sum(dim=1, keepdim=True), rtol=0, atol=0
    )
    native.sum().backward()
    z_full.sum().backward()
    torch.testing.assert_close(decomposed_input.grad, native_input.grad, rtol=0, atol=0)


def test_capture_reads_existing_fusion_without_changing_output_or_parameters() -> None:
    class ToyFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.outconv = nn.Conv2d(6, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.outconv(x)

    model = ToyFusion()
    before_keys = tuple(model.state_dict())
    before_parameters = sum(parameter.numel() for parameter in model.parameters())
    x = torch.randn(1, 6, 3, 3)
    with PointwiseFusionCapture(model.outconv) as capture:
        native = model(x)
        z_full, contributions, _ = capture.decompose()

    torch.testing.assert_close(z_full, native, rtol=0, atol=0)
    assert contributions.shape == (1, 6, 3, 3)
    assert tuple(model.state_dict()) == before_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == before_parameters
    assert len(model.outconv._forward_pre_hooks) == 0


def test_sdrr_accepts_six_native_contributions() -> None:
    z_full = torch.ones(1, 1, 1, 2)
    contributions = torch.zeros(1, 6, 1, 2, requires_grad=True)
    with torch.no_grad():
        contributions[0, 4, 0, 0] = 2.0
    target = torch.zeros_like(z_full)

    loss, logs = counterfactual_responsibility_suppression(
        z_full, contributions, target, safe_kernel=1
    )
    loss.backward()

    assert logs["responsible_count"] == 1
    assert contributions.grad is not None
    assert int((contributions.grad != 0).sum()) == 1
    assert contributions.grad[0, 4, 0, 0] != 0
