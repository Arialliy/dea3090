from __future__ import annotations

import subprocess

import torch

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.baselines.mshnet_official import MSHNet as CanonicalMSHNet


SOURCE_COMMIT = "46cdfd46802629da51f70124662af7335be74b56"


def historical_mshnet_class():
    source = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:model/MSHNet.py"], text=True
    )
    namespace = {"__name__": "_historical_mshnet"}
    exec(compile(source, f"{SOURCE_COMMIT}:model/MSHNet.py", "exec"), namespace)
    return namespace["MSHNet"]


def test_canonical_baseline_is_forward_and_parameter_identical_to_source_commit():
    historical_type = historical_mshnet_class()

    torch.manual_seed(3107)
    historical = historical_type(3).eval()
    torch.manual_seed(3107)
    canonical = CanonicalMSHNet(3).eval()

    assert historical.state_dict().keys() == canonical.state_dict().keys()
    for name, expected in historical.state_dict().items():
        torch.testing.assert_close(
            canonical.state_dict()[name], expected, rtol=0.0, atol=0.0
        )

    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        for warm_flag in (False, True):
            historical_sides, historical_final = historical(images, warm_flag)
            canonical_sides, canonical_final = canonical(images, warm_flag)
            assert len(historical_sides) == len(canonical_sides)
            for actual, expected in zip(canonical_sides, historical_sides):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                canonical_final, historical_final, rtol=0.0, atol=0.0
            )


def test_deterministic_variant_preserves_the_locked_forward_and_state_schema():
    torch.manual_seed(3108)
    canonical = CanonicalMSHNet(3).eval()
    deterministic = DeterministicMSHNet(3).eval()
    deterministic.load_state_dict(canonical.state_dict(), strict=True)

    assert canonical.state_dict().keys() == deterministic.state_dict().keys()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        canonical_sides, canonical_final = canonical(images, True)
        deterministic_sides, deterministic_final = deterministic(images, True)
    for actual, expected in zip(deterministic_sides, canonical_sides):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        deterministic_final, canonical_final, rtol=0.0, atol=0.0
    )
