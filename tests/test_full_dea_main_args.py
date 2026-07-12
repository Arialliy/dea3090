from __future__ import annotations

import os
import sys
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import Trainer, get_method_metadata, get_method_name, get_run_folder_name, validate_args
from model.MSHNet import MSHNet
from model.dea_mshnet import DEAMSHNet
from model.full_dea_mshnet import FullDEAMSHNet
from model.loss import SLSIoULoss


def make_args(**kwargs):
    args = Namespace(
        model_type="mshnet",
        init_from_baseline="",
        if_checkpoint=False,
        dea_lambda_single=0.0,
        dea_lambda_dec=0.0,
        dea_lambda_empty=0.0,
        full_dea_lambda=1.0,
        full_dea_version="v3",
        full_dea_ramp_epochs=30,
        full_dea_start_epoch=0,
        full_dea_freeze_backbone_epochs=0,
        full_dea_tau_base=0.45,
        full_dea_tau_target=0.45,
        full_dea_tau_scale=0.45,
        full_dea_topk_ratio=0.001,
        full_dea_topk_min_score=0.45,
        full_dea_max_hard_bg_ratio=0.003,
        full_dea_safe_kernel=15,
        full_dea_protect_kernel=9,
        full_dea_hard_min_area=1,
        full_dea_hard_max_area=256,
        integrated_route_channels=16,
        integrated_route_temperature=1.0,
        integrated_routing_mode="dea",
        integrated_decoder_routing=True,
        integrated_scale_routing=True,
        integrated_route_upsample_mode="nearest-exact",
        integrated_update_limit=0.25,
        integrated_uncertain_margin=1.0,
        integrated_route_loss_weight=0.05,
        integrated_route_ramp_epochs=3,
        integrated_isolate_route_gradients=True,
        predictive_state_channels=32,
        predictive_step_size=1.0,
        predictive_delta_init=1.0,
        predictive_delta_min=0.05,
        predictive_legacy_numerics=False,
        predictive_log_interval=50,
        dataset_dir="datasets/NUAA-SIRST",
        seed=20260706,
        deterministic=True,
        s3_start_epoch=20,
        hms_ramp_epochs=20,
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def test_full_dea_rejects_dea_lite_lambdas() -> None:
    args = make_args(model_type="full_dea", dea_lambda_single=0.01)
    with pytest.raises(ValueError):
        validate_args(args)


def test_full_dea_rejects_invalid_safe_kernel() -> None:
    args = make_args(model_type="full_dea", full_dea_safe_kernel=14)
    with pytest.raises(ValueError):
        validate_args(args)


def test_method_metadata_names_full_dea_v3() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_method_name(args) == "FullDEA-v3-TPS"
    meta = get_method_metadata(args)
    assert meta["model_type"] == "full_dea"
    assert meta["method"] == "FullDEA-v3-TPS"
    assert meta["full_dea_version"] == "v3"


def test_method_metadata_can_name_full_dea_v2_for_audit() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v2"))
    assert get_method_name(args) == "FullDEA-v2"


def test_method_metadata_names_full_dea_v4_relation_reasoner() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v4"))
    assert get_method_name(args) == "FullDEA-v4-CRR"
    meta = get_method_metadata(args)
    assert meta["full_dea_version"] == "v4"


def test_method_metadata_names_full_dea_v5_hard_transport() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v5"))
    assert get_method_name(args) == "FullDEA-v5-CRR-HT"


def test_run_folder_name_uses_method_name() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_run_folder_name(args, "2026-07-09-22-00-00") == (
        "FullDEA-v3-TPS-2026-07-09-22-00-00"
    )


def test_method_metadata_persists_run_label() -> None:
    args = validate_args(make_args(run_label="nuaa_seed_11"))
    assert get_method_metadata(args)["run_label"] == "nuaa_seed_11"


def test_rods_method_metadata_and_instance_map_flag() -> None:
    args = validate_args(make_args(
        deep_supervision="rods_interval",
        aux_loss_weight=0.8,
        ownership_preferred_cells=3.0,
        ownership_sigma=0.75,
        ownership_min_decidability=0.25,
        ownership_interval_ratio=0.5,
        ownership_fallback="side0",
        ownership_ignore_dilation=3,
        empty_side_policy="skip",
    ))

    assert get_method_name(args) == "RODS-Interval"
    assert args.return_instance_map is True
    metadata = get_method_metadata(args)
    assert metadata["method"] == "RODS-Interval"
    assert metadata["deep_supervision"] == "rods_interval"
    assert metadata["ownership_sigma"] == 0.75


def test_rods_rejects_dea_lite_lambdas() -> None:
    with pytest.raises(ValueError, match="must not be mixed"):
        validate_args(make_args(
            deep_supervision="rods_hard",
            dea_lambda_single=0.01,
        ))


def test_tfds_projection_uses_honest_method_name_and_instance_maps() -> None:
    args = validate_args(make_args(
        deep_supervision="tfds_projection",
        tfds_min_iou=0.5,
        tfds_max_centroid_distance=3.0,
    ))

    assert get_method_name(args) == "TCDS-Projection"
    assert args.return_instance_map is True
    metadata = get_method_metadata(args)
    assert metadata["deep_supervision"] == "tfds_projection"
    assert metadata["tfds_min_iou"] == 0.5
    assert metadata["tfds_max_centroid_distance"] == 3.0


def test_tfds_projection_rejects_invalid_task_thresholds() -> None:
    with pytest.raises(ValueError, match="tfds-min-iou"):
        validate_args(make_args(
            deep_supervision="tfds_projection",
            tfds_min_iou=1.1,
        ))


def test_tfds_active_renorm_is_explicitly_named_as_diagnostic() -> None:
    args = validate_args(make_args(
        deep_supervision="tfds_projection_active_renorm",
    ))

    assert get_method_name(args) == "TCDS-Projection-ActiveRenorm"
    assert args.return_instance_map is True


def test_tgds_halfspace_is_parameter_free_deep_supervision() -> None:
    args = validate_args(make_args(deep_supervision="tgds_halfspace"))

    assert get_method_name(args) == "TGDS-Halfspace"
    assert args.return_instance_map is False
    metadata = get_method_metadata(args)
    assert metadata["deep_supervision"] == "tgds_halfspace"


def test_tsds_lift_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="tsds_lift"))

    assert get_method_name(args) == "TSDS-Lift"
    assert args.return_instance_map is False
    assert get_method_metadata(args)["deep_supervision"] == "tsds_lift"


def test_prds_regret_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="prds_regret"))

    assert get_method_name(args) == "PRDS-Regret"
    assert args.return_instance_map is False
    assert get_method_metadata(args)["deep_supervision"] == "prds_regret"


def test_cscs_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="cscs_leave_one_out"))

    assert get_method_name(args) == "CSCS-LeaveOneOut"
    assert args.return_instance_map is False
    assert get_method_metadata(args)["deep_supervision"] == (
        "cscs_leave_one_out"
    )


def test_sfds_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="sfds_filtration"))

    assert get_method_name(args) == "SFDS-Filtration"
    assert args.return_instance_map is False
    assert get_method_metadata(args)["deep_supervision"] == "sfds_filtration"


def test_asfs_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="asfs_anchor_filtration"))

    assert get_method_name(args) == "ASFS-AnchorFiltration"
    assert args.return_instance_map is False
    assert get_method_metadata(args)["deep_supervision"] == (
        "asfs_anchor_filtration"
    )


def test_rdfs_persists_role_discovery_schedule() -> None:
    args = validate_args(make_args(
        deep_supervision="rdfs_continuation",
        rdfs_start_epoch=20,
        rdfs_ramp_epochs=15,
    ))

    assert get_method_name(args) == "RDFS-Continuation"
    metadata = get_method_metadata(args)
    assert metadata["rdfs_start_epoch"] == 20
    assert metadata["rdfs_ramp_epochs"] == 15

    with pytest.raises(ValueError, match="rdfs-start-epoch"):
        validate_args(make_args(
            deep_supervision="rdfs_continuation",
            rdfs_start_epoch=-1,
        ))
    with pytest.raises(ValueError, match="rdfs-ramp-epochs"):
        validate_args(make_args(
            deep_supervision="rdfs_continuation",
            rdfs_ramp_epochs=0,
        ))


def test_crs_persists_counterfactual_constraint_semantics() -> None:
    args = validate_args(make_args(
        deep_supervision="crs_flip_suppression",
        crs_lambda=0.05,
        crs_start_epoch=20,
        crs_ramp_epochs=10,
        crs_safe_kernel=15,
        crs_detach_scale_evidence=True,
    ))

    assert get_method_name(args) == "SDRR-ScaleDeletionResponsibility"
    assert args.return_instance_map is False
    metadata = get_method_metadata(args)
    assert metadata["crs_lambda"] == 0.05
    assert metadata["crs_start_epoch"] == 20
    assert metadata["crs_ramp_epochs"] == 10
    assert metadata["crs_safe_kernel"] == 15
    assert metadata["crs_detach_scale_evidence"] is True
    assert metadata["sdrr_normalization"] == "event"

    with pytest.raises(ValueError, match="crs-safe-kernel"):
        validate_args(make_args(
            deep_supervision="crs_flip_suppression",
            crs_safe_kernel=8,
        ))


def test_sdrr_normalization_controls_are_named_and_validated() -> None:
    args = validate_args(
        make_args(
            deep_supervision="crs_flip_suppression",
            sdrr_normalization="unique_pixel",
        )
    )
    assert get_method_name(args) == "SDRR-NormalizationControl-unique_pixel"
    assert get_method_metadata(args)["sdrr_normalization"] == "unique_pixel"

    with pytest.raises(ValueError, match="sdrr-normalization"):
        validate_args(
            make_args(
                deep_supervision="crs_flip_suppression",
                sdrr_normalization="invalid",
            )
        )


def test_crs_matched_random_is_explicit_ablation_not_main_method() -> None:
    args = validate_args(make_args(
        deep_supervision="crs_matched_random",
        crs_lambda=0.05,
        crs_start_epoch=250,
        crs_ramp_epochs=50,
        crs_safe_kernel=15,
    ))

    assert get_method_name(args) == "SDRR-ScaleBudgetRandomControl-Unmatched"
    assert args.return_instance_map is False
    metadata = get_method_metadata(args)
    assert metadata["deep_supervision"] == "crs_matched_random"
    assert metadata["crs_start_epoch"] == 250


def test_same_pixel_random_scale_control_is_explicit() -> None:
    args = validate_args(
        make_args(deep_supervision="crs_same_pixel_random_scale")
    )
    assert get_method_name(args) == "SDRR-SamePixelRandomScaleControl"
    assert args.return_instance_map is False


def test_magnitude_nonpivotal_control_is_explicit() -> None:
    args = validate_args(make_args(deep_supervision="crs_magnitude_nonpivotal"))
    assert (
        get_method_name(args)
        == "SDRR-MagnitudeMatchedNonPivotalControl"
    )
    assert args.return_instance_map is False


def test_clean_mshnet_variants_are_physically_explicit() -> None:
    official = validate_args(make_args(mshnet_variant="official"))
    deterministic = validate_args(make_args(mshnet_variant="deterministic"))

    assert get_method_name(official) == "MSHNet-OfficialForward"
    assert get_method_name(deterministic) == "MSHNet-Deterministic"
    assert get_method_metadata(official)["mshnet_variant"] == "official"
    assert get_method_metadata(deterministic)["mshnet_variant"] == "deterministic"


def test_clean_mshnet_rejects_workbench_only_forward_switches() -> None:
    with pytest.raises(ValueError, match="DEA-lite"):
        validate_args(
            make_args(mshnet_variant="deterministic", dea_lambda_single=0.1)
        )
    with pytest.raises(ValueError, match="fusion_alpha"):
        validate_args(
            make_args(
                mshnet_variant="deterministic",
                deep_supervision="hms_continuation",
            )
        )


@pytest.mark.parametrize(
    ("mode", "name"),
    [
        ("legacy_no_s3", "MSHNet-Control-NoS3"),
        ("legacy_no_s2s3", "MSHNet-Control-NoS2S3"),
        ("legacy_s0_only", "MSHNet-Control-S0Only"),
    ],
)
def test_scale_subset_controls_are_explicit(mode: str, name: str) -> None:
    args = validate_args(make_args(deep_supervision=mode))
    assert get_method_name(args) == name
    assert args.return_instance_map is False


def test_scale_subset_warmup_without_side_logits_degenerates_to_final() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.args = make_args(deep_supervision="legacy_no_s3")
    trainer.loss_fun = SLSIoULoss()
    trainer.warm_epoch = 5
    trainer.down = nn.MaxPool2d(2, 2)
    trainer.last_deep_supervision_log = {}
    trainer.last_tgds_components = None
    prediction = torch.zeros(2, 1, 8, 8, requires_grad=True)
    labels = torch.zeros_like(prediction)

    actual = trainer.compute_deep_supervision_loss(
        prediction,
        [],
        labels,
        instance_map=None,
        epoch=0,
    )
    expected = trainer.loss_fun(prediction, labels, trainer.warm_epoch, 0)
    torch.testing.assert_close(actual, expected)


def test_s3_delayed_control_is_explicit_and_validates_start_epoch() -> None:
    args = validate_args(make_args(
        deep_supervision="legacy_s3_delayed",
        s3_start_epoch=20,
    ))
    assert get_method_name(args) == "MSHNet-Control-S3Delayed"
    assert get_method_metadata(args)["s3_start_epoch"] == 20
    with pytest.raises(ValueError, match="s3-start-epoch"):
        validate_args(make_args(
            deep_supervision="legacy_s3_delayed",
            s3_start_epoch=-1,
        ))


def test_hms_continuation_is_parameter_free_and_validates_ramp() -> None:
    args = validate_args(make_args(
        deep_supervision="hms_continuation",
        hms_ramp_epochs=20,
    ))
    assert get_method_name(args) == "MSHNet-HMS-Continuation"
    assert get_method_metadata(args)["hms_ramp_epochs"] == 20
    with pytest.raises(ValueError, match="hms-ramp-epochs"):
        validate_args(make_args(
            deep_supervision="hms_continuation",
            hms_ramp_epochs=0,
        ))


def test_mcsls_is_parameter_free_and_needs_no_instance_map() -> None:
    args = validate_args(make_args(deep_supervision="mcsls_null_safe"))
    assert get_method_name(args) == "MSHNet-MC-SLS"
    assert args.return_instance_map is False


def test_zmsls_abstention_is_explicit_and_parameter_free() -> None:
    args = validate_args(make_args(deep_supervision="zmsls_null_abstain"))
    assert get_method_name(args) == "MSHNet-ZM-SLS-Abstain"
    assert args.return_instance_map is False


def test_rods_checkpoint_metadata_rejects_ownership_mismatch() -> None:
    args = validate_args(make_args(
        deep_supervision="rods_interval",
        aux_loss_weight=0.8,
        ownership_preferred_cells=3.0,
        ownership_sigma=0.75,
        ownership_min_decidability=0.25,
        ownership_interval_ratio=0.5,
        ownership_fallback="side0",
        ownership_ignore_dilation=3,
        empty_side_policy="skip",
    ))
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    metadata = get_method_metadata(args)

    Trainer.validate_integrated_checkpoint_metadata(
        trainer, {"method_meta": metadata}, check_split_hashes=False
    )

    incompatible = dict(metadata)
    incompatible["ownership_sigma"] = 0.5
    with pytest.raises(RuntimeError, match="ownership_sigma"):
        Trainer.validate_integrated_checkpoint_metadata(
            trainer, {"method_meta": incompatible}, check_split_hashes=False
        )


def test_integrated_method_name_exposes_residual_alignment() -> None:
    args = validate_args(make_args(model_type="dea_integrated"))
    assert get_method_name(args) == "DEAIntegrated-ResidualAligned"


def test_integrated_rejects_route_loss_for_attention_control() -> None:
    args = make_args(
        model_type="dea_integrated",
        integrated_routing_mode="attention",
        integrated_route_loss_weight=0.05,
    )
    with pytest.raises(ValueError, match="not defined for the attention"):
        validate_args(args)


def test_integrated_rejects_nonexclusive_hard_gate_interpolation() -> None:
    args = make_args(
        model_type="dea_integrated",
        integrated_route_upsample_mode="bilinear",
    )
    with pytest.raises(ValueError, match="Hard scale routing"):
        validate_args(args)


def test_dea_main_method_name_exposes_state_width() -> None:
    args = validate_args(make_args(model_type="dea"))
    assert get_method_name(args) == "DEA-v0-C32-Eta1"
    metadata = get_method_metadata(args)
    assert metadata["dea_state_channels"] == 32
    assert metadata["dea_step_size"] == 1.0
    assert metadata["dea_legacy_numerics"] is False

    half_step = validate_args(make_args(
        model_type="dea",
        predictive_step_size=0.5,
        predictive_legacy_numerics=True,
    ))
    assert get_method_name(half_step) == (
        "DEA-v0-C32-Eta0p5-LegacyNum"
    )
    compatibility_alias = validate_args(make_args(
        model_type="predictive_correction"
    ))
    assert get_method_name(compatibility_alias) == (
        "PredictiveCorrection-C32-Eta1"
    )


def test_dea_checkpoint_metadata_rejects_numerics_mismatch() -> None:
    args = validate_args(make_args(model_type="dea"))
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    metadata = get_method_metadata(args)

    Trainer.validate_integrated_checkpoint_metadata(
        trainer, {"method_meta": metadata}, check_split_hashes=False
    )
    incompatible = dict(metadata)
    incompatible["dea_legacy_numerics"] = True
    with pytest.raises(RuntimeError, match="legacy_numerics"):
        Trainer.validate_integrated_checkpoint_metadata(
            trainer, {"method_meta": incompatible}, check_split_hashes=False
        )
    missing_field = dict(metadata)
    missing_field.pop("dea_legacy_numerics")
    with pytest.raises(RuntimeError, match="<missing>"):
        Trainer.validate_integrated_checkpoint_metadata(
            trainer, {"method_meta": missing_field}, check_split_hashes=False
        )


def test_dea_partial_load_accepts_only_replaced_decoder_keys() -> None:
    torch.manual_seed(11)
    baseline = MSHNet(3)
    predictive = DEAMSHNet(3, state_channels=32)
    trainer = Trainer.__new__(Trainer)
    trainer.model = predictive

    Trainer.load_model_state_partial(
        trainer,
        baseline.state_dict(),
        allowed_missing_prefixes=DEAMSHNet.BASELINE_MISSING_PREFIXES,
        allowed_unexpected_prefixes=DEAMSHNet.BASELINE_UNEXPECTED_PREFIXES,
    )

    assert torch.equal(
        predictive.conv_init.weight, baseline.conv_init.weight
    )
    assert torch.equal(
        predictive.encoder_3[1].conv2.weight,
        baseline.encoder_3[1].conv2.weight,
    )


def test_dea_main_rejects_invalid_dynamics() -> None:
    with pytest.raises(ValueError, match="state-channels"):
        validate_args(make_args(
            model_type="dea",
            predictive_state_channels=1,
        ))
    with pytest.raises(ValueError, match="step-size"):
        validate_args(make_args(
            model_type="dea",
            predictive_step_size=1.5,
        ))
    with pytest.raises(ValueError, match="delta-init"):
        validate_args(make_args(
            model_type="dea",
            predictive_delta_init=0.01,
        ))


def test_frozen_backbone_keeps_batchnorm_statistics_fixed() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.args = Namespace(
        model_type="full_dea",
        full_dea_freeze_backbone_epochs=2,
    )
    trainer.model = FullDEAMSHNet(input_channels=3, full_dea_version="v3")
    trainer.model.train()

    Trainer.configure_full_dea_trainable(trainer, epoch=0)

    for name, parameter in trainer.model.named_parameters():
        assert parameter.requires_grad == name.startswith("full_dea_head")
    for name, module in trainer.model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert not module.training, name
    assert trainer.model.full_dea_head.training
