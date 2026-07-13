from __future__ import annotations

import numpy as np
import torch

from tools.audit_csl_stage0 import csl_statistic_maps, region_scores


def test_csl_audit_maps_are_finite_and_region_addressable() -> None:
    e0 = torch.zeros(1, 2, 4, 4)
    e0[:, :, 0, 0] = 4.0
    maps = csl_statistic_maps(e0, (4, 4))
    assert all(value.shape == (4, 4) for value in maps.values())
    assert all(np.isfinite(value).all() for value in maps.values())
    scores = region_scores(maps, np.asarray([[0, 0], [0, 1]]))
    assert set(scores) == set(maps)
    assert scores["exclusive_ratio"] > 0.0
