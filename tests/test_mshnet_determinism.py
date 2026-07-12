from __future__ import annotations

import torch

from main import seed_everything
from model.MSHNet import ChannelAttention


def test_channel_attention_amax_is_forward_equivalent_to_adaptive_pool() -> None:
    torch.manual_seed(41)
    module = ChannelAttention(16).eval()
    value = torch.randn(3, 16, 9, 7)

    with torch.no_grad():
        avg_out = module.fc2(module.relu1(module.fc1(module.avg_pool(value))))
        pooled_max = module.max_pool(value)
        max_out = module.fc2(module.relu1(module.fc1(pooled_max)))
        expected = module.sigmoid(avg_out + max_out)
        actual = module(value)

    torch.testing.assert_close(
        torch.amax(value, dim=(-2, -1), keepdim=True),
        pooled_max,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_seed_everything_enables_fail_closed_determinism() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        seed_everything(43, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
