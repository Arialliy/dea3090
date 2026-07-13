from __future__ import annotations

import torch

from model.sdr_mshnet import SDRMSHNet
from tools.package_sdr_mshnet_run import _strict_identity


def test_strict_identity_accepts_complete_sdr_schema() -> None:
    state = SDRMSHNet(3).state_dict()
    report = _strict_identity(state)

    assert report["strict_state_load"] is True
    assert report["default_forward_bit_exact"] is True
    assert report["added_parameter_elements"] == 0
    assert report["parameter_elements"] == 4_065_513
    assert all(torch.isfinite(value).all() for value in state.values())
