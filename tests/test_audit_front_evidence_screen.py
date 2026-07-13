from __future__ import annotations

import numpy as np
import torch

from tools.audit_front_evidence_screen import front_statistic_maps


def test_front_screen_maps_are_finite_and_shape_aligned() -> None:
    torch.manual_seed(81)
    image = torch.randn(1, 3, 8, 8)
    e0 = torch.relu(torch.randn(1, 16, 8, 8))
    maps = front_statistic_maps(image, e0, (8, 8))
    assert len(maps) >= 10
    assert all(value.shape == (8, 8) for value in maps.values())
    assert all(np.isfinite(value).all() for value in maps.values())
