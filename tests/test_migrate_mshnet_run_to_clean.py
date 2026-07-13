from __future__ import annotations

from torch.optim import Adagrad

from model.MSHNet import MSHNet as WorkbenchMSHNet
from tools.migrate_mshnet_run_to_clean import clean_checkpoint_payload


def test_clean_checkpoint_migration_removes_only_zero_dormant_state() -> None:
    model = WorkbenchMSHNet(3)
    optimizer = Adagrad(model.parameters(), lr=0.05)
    payload = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "method_meta": {
            "method": "MSHNet",
            "deep_supervision": "legacy_exact",
        },
    }

    migrated, report = clean_checkpoint_payload(
        payload, run_label="clean_test"
    )

    assert report["removed_parameter_elements"] == 521
    assert report["strict_model_and_optimizer_load"]
    assert not any(
        name.startswith("decidability_head.") for name in migrated["net"]
    )
    assert len(migrated["optimizer"]["param_groups"][0]["params"]) == 220
    assert migrated["method_meta"]["mshnet_variant"] == "deterministic"
    assert migrated["method_meta"]["method"] == "MSHNet-Deterministic"
