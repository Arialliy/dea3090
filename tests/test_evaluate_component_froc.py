from __future__ import annotations

import pytest
import torch

from tools.evaluate_component_froc import MODEL_VARIANTS, extract_state_dict


def test_component_froc_tool_exposes_only_audited_variants() -> None:
    assert set(MODEL_VARIANTS) == {"deterministic", "ccfd", "spt0"}


def test_extract_state_dict_accepts_checkpoint_and_removes_data_parallel_prefix() -> None:
    tensor = torch.ones(1)
    state = extract_state_dict({"net": {"module.final.weight": tensor}})
    assert set(state) == {"final.weight"}
    assert state["final.weight"] is tensor


def test_extract_state_dict_fails_closed_on_non_tensor_payload() -> None:
    with pytest.raises(ValueError, match="non-tensor"):
        extract_state_dict({"net": {"bad": "value"}})
