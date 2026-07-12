from utils.data import *
from utils.metric import *
from argparse import ArgumentParser, ArgumentTypeError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from model.MSHNet import *
from model.baselines.mshnet_deterministic import (
    MSHNet as DeterministicBaselineMSHNet,
)
from model.baselines.mshnet_official import MSHNet as OfficialBaselineMSHNet
from model.loss import *
from model.full_dea_mshnet import FullDEAMSHNet
from model.dea_integrated_mshnet import DEAIntegratedMSHNet
from model.dea_mshnet import DEAMSHNet
from model.dea_counterfactual_veto import (
    CounterfactualVetoMSHNet,
    FineScaleVetoMSHNet,
    SharedCEVMSHNet,
)
from model.dea_integrated_loss import residual_aligned_route_loss
from model.full_dea_loss import (
    full_dea_aux_loss_v2,
    full_dea_aux_loss_v3,
    full_dea_aux_loss_v4,
)
from model.masked_owned_loss import MaskedOwnedScaleIoULoss
from model.resolution_owned_supervision import (
    OwnedSideSupervisionBuilder,
    ResolutionDecidableSupervisionGraph,
)
from model.partial_sls_loss import PartialSLSIoULoss
from model.measure_conditioned_sls import MeasureConditionedSLSIoULoss
from model.scale_coalition_supervision import (
    leave_one_scale_out_coalitions,
    nested_scale_filtration,
)
from model.counterfactual_responsibility import (
    counterfactual_responsibility_suppression,
    matched_random_responsibility_suppression,
    magnitude_matched_nonpivotal_suppression,
    same_pixel_random_scale_suppression,
)
from model.task_gradient_supervision import (
    combine_task_and_auxiliary_gradients,
    gradient_inner_product,
    project_auxiliary_gradient,
)
from model.task_consistent_supervision import (
    TaskConsistentPartialTargetBuilder,
    TaskConsistentProjectionGraph,
)
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
import glob
import random
import numpy as np
import json

PROJECT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_DATASET_DIR = osp.join(PROJECT_DIR, 'datasets', 'IRSTD-1K')
DEFAULT_WEIGHT_DIR = osp.join(PROJECT_DIR, 'weight')
DEA_MODEL_TYPES = ('dea', 'predictive_correction')
CEV_CONTROL_TYPES = ('dea_fine_veto_control', 'dea_cev_control')
RODS_DEEP_SUPERVISION_TYPES = (
    'rods_interval',
    'rods_hard',
    'rods_random',
    'rods_area_only',
)
TFDS_DEEP_SUPERVISION_TYPES = (
    'tfds_projection',
    'tfds_projection_active_renorm',
)
TGDS_DEEP_SUPERVISION_TYPES = (
    'tgds_halfspace',
)
TSDS_DEEP_SUPERVISION_TYPES = (
    'tsds_lift',
)
PRDS_DEEP_SUPERVISION_TYPES = (
    'prds_regret',
)
COALITION_DEEP_SUPERVISION_TYPES = (
    'cscs_leave_one_out',
    'sfds_filtration',
    'asfs_anchor_filtration',
    'rdfs_continuation',
)
RESPONSIBILITY_DEEP_SUPERVISION_TYPES = (
    'crs_flip_suppression',
    'crs_matched_random',
    'crs_same_pixel_random_scale',
    'crs_magnitude_nonpivotal',
)
SCALE_SUBSET_DEEP_SUPERVISION = {
    'legacy_no_s3': (0, 1, 2),
    'legacy_no_s2s3': (0, 1),
    'legacy_s0_only': (0,),
}
SCHEDULED_DEEP_SUPERVISION_TYPES = (
    'legacy_s3_delayed',
)
HOMOTOPY_DEEP_SUPERVISION_TYPES = (
    'hms_continuation',
)
MEASURE_CONDITIONED_DEEP_SUPERVISION_TYPES = (
    'mcsls_null_safe',
    'zmsls_null_abstain',
)


def is_dea_main_model(model_type):
    return model_type in DEA_MODEL_TYPES

def is_cev_control(model_type):
    return model_type in CEV_CONTROL_TYPES

def is_rods_deep_supervision(deep_supervision):
    return deep_supervision in RODS_DEEP_SUPERVISION_TYPES

def is_tfds_deep_supervision(deep_supervision):
    return deep_supervision in TFDS_DEEP_SUPERVISION_TYPES

def is_tgds_deep_supervision(deep_supervision):
    return deep_supervision in TGDS_DEEP_SUPERVISION_TYPES

def is_tsds_deep_supervision(deep_supervision):
    return deep_supervision in TSDS_DEEP_SUPERVISION_TYPES

def is_prds_deep_supervision(deep_supervision):
    return deep_supervision in PRDS_DEEP_SUPERVISION_TYPES

def is_coalition_deep_supervision(deep_supervision):
    return deep_supervision in COALITION_DEEP_SUPERVISION_TYPES

def is_responsibility_deep_supervision(deep_supervision):
    return deep_supervision in RESPONSIBILITY_DEEP_SUPERVISION_TYPES

def is_scale_subset_deep_supervision(deep_supervision):
    return deep_supervision in SCALE_SUBSET_DEEP_SUPERVISION

def is_scheduled_deep_supervision(deep_supervision):
    return deep_supervision in SCHEDULED_DEEP_SUPERVISION_TYPES

def is_homotopy_deep_supervision(deep_supervision):
    return deep_supervision in HOMOTOPY_DEEP_SUPERVISION_TYPES

def is_measure_conditioned_deep_supervision(deep_supervision):
    return deep_supervision in MEASURE_CONDITIONED_DEEP_SUPERVISION_TYPES

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise ArgumentTypeError('Boolean value expected.')

def load_torch_file(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

def get_dea_ramp(epoch, warm_epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warm_epoch:
        return 0.0
    return min(1.0, float(epoch - warm_epoch) / float(ramp_epochs))

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def validate_args(args):
    if getattr(args, "mode", "train") not in ("train", "test"):
        raise ValueError("--mode must be train or test.")
    if not hasattr(args, "deep_supervision"):
        args.deep_supervision = "legacy_exact"
    if args.deep_supervision == "legacy":
        args.deep_supervision = "legacy_exact"
    deep_supervision_choices = (
        "legacy_exact",
        "legacy_rescaled",
        "final_only",
        "side_no_location",
    ) + RODS_DEEP_SUPERVISION_TYPES + TFDS_DEEP_SUPERVISION_TYPES + TGDS_DEEP_SUPERVISION_TYPES + TSDS_DEEP_SUPERVISION_TYPES + PRDS_DEEP_SUPERVISION_TYPES + COALITION_DEEP_SUPERVISION_TYPES + RESPONSIBILITY_DEEP_SUPERVISION_TYPES + tuple(SCALE_SUBSET_DEEP_SUPERVISION) + SCHEDULED_DEEP_SUPERVISION_TYPES + HOMOTOPY_DEEP_SUPERVISION_TYPES + MEASURE_CONDITIONED_DEEP_SUPERVISION_TYPES
    if args.deep_supervision not in deep_supervision_choices:
        raise ValueError("--deep-supervision has an unsupported value.")
    defaults = {
        "aux_loss_weight": 0.8,
        "ownership_preferred_cells": 3.0,
        "ownership_sigma": 0.75,
        "ownership_min_decidability": 0.25,
        "ownership_interval_ratio": 0.5,
        "ownership_fallback": "side0",
        "ownership_ignore_dilation": 3,
        "empty_side_policy": "skip",
        "rods_log_interval": 50,
        "tfds_min_iou": 0.5,
        "tfds_max_centroid_distance": 3.0,
        "s3_start_epoch": 20,
        "hms_ramp_epochs": 20,
        "rdfs_start_epoch": 20,
        "rdfs_ramp_epochs": 20,
        "crs_lambda": 0.05,
        "crs_start_epoch": 250,
        "crs_ramp_epochs": 50,
        "crs_safe_kernel": 15,
        "crs_detach_scale_evidence": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.deep_supervision != "legacy_exact" and args.model_type != "mshnet":
        raise ValueError(
            "--deep-supervision modes other than legacy_exact are currently "
            "implemented only for --model-type mshnet."
        )
    if args.deep_supervision != "legacy_exact":
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "RODS/alternative deep supervision must not be mixed with "
                "DEA-lite losses."
            )
    if float(getattr(args, "aux_loss_weight", 0.8)) < 0.0:
        raise ValueError("--aux-loss-weight must be non-negative.")
    if float(getattr(args, "ownership_sigma", 0.75)) <= 0.0:
        raise ValueError("--ownership-sigma must be positive.")
    if float(getattr(args, "ownership_preferred_cells", 3.0)) <= 0.0:
        raise ValueError("--ownership-preferred-cells must be positive.")
    if float(getattr(args, "ownership_interval_ratio", 0.5)) < 0.0:
        raise ValueError("--ownership-interval-ratio must be non-negative.")
    if args.ownership_fallback not in ("side0", "final_only"):
        raise ValueError("--ownership-fallback must be side0 or final_only.")
    if args.empty_side_policy not in ("skip", "background_only"):
        raise ValueError("--empty-side-policy must be skip or background_only.")
    if int(getattr(args, "rods_log_interval", 50)) < 0:
        raise ValueError("--rods-log-interval must be non-negative.")
    if not 0.0 <= float(args.tfds_min_iou) <= 1.0:
        raise ValueError("--tfds-min-iou must be in [0, 1].")
    if float(args.tfds_max_centroid_distance) <= 0.0:
        raise ValueError("--tfds-max-centroid-distance must be positive.")
    if int(args.s3_start_epoch) < 0:
        raise ValueError("--s3-start-epoch must be non-negative.")
    if int(args.hms_ramp_epochs) < 1:
        raise ValueError("--hms-ramp-epochs must be >= 1.")
    if int(args.rdfs_start_epoch) < 0:
        raise ValueError("--rdfs-start-epoch must be >= 0.")
    if int(args.rdfs_ramp_epochs) < 1:
        raise ValueError("--rdfs-ramp-epochs must be >= 1.")
    if float(args.crs_lambda) < 0.0:
        raise ValueError("--crs-lambda must be non-negative.")
    if int(args.crs_start_epoch) < 0:
        raise ValueError("--crs-start-epoch must be >= 0.")
    if int(args.crs_ramp_epochs) < 1:
        raise ValueError("--crs-ramp-epochs must be >= 1.")
    if int(args.crs_safe_kernel) <= 0 or int(args.crs_safe_kernel) % 2 == 0:
        raise ValueError("--crs-safe-kernel must be a positive odd integer.")
    if getattr(args, "sdrr_normalization", "event") not in (
        "event",
        "safe_density",
        "unique_pixel",
    ):
        raise ValueError(
            "--sdrr-normalization must be event, safe_density, or unique_pixel."
        )
    mshnet_variant = getattr(args, "mshnet_variant", "workbench")
    if mshnet_variant not in ("workbench", "official", "deterministic"):
        raise ValueError(
            "--mshnet-variant must be workbench, official, or deterministic."
        )
    if args.model_type == "mshnet" and mshnet_variant != "workbench":
        if any(
            float(value) != 0.0
            for value in (
                args.dea_lambda_single,
                args.dea_lambda_dec,
                args.dea_lambda_empty,
            )
        ):
            raise ValueError(
                "clean MSHNet variants cannot expose the legacy DEA-lite head"
            )
        if is_homotopy_deep_supervision(args.deep_supervision):
            raise ValueError(
                "clean MSHNet variants do not expose legacy fusion_alpha"
            )
    if int(getattr(args, "ownership_ignore_dilation", 3)) <= 0 or (
        int(getattr(args, "ownership_ignore_dilation", 3)) % 2 == 0
    ):
        raise ValueError("--ownership-ignore-dilation must be a positive odd integer.")
    args.return_instance_map = (
        is_rods_deep_supervision(args.deep_supervision)
        or is_tfds_deep_supervision(args.deep_supervision)
    )

    if args.model_type == "dea_integrated":
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError("--if-checkpoint and --init-from-baseline are separate paths.")
        if not args.init_from_baseline and not args.if_checkpoint and getattr(args, "mode", "train") == "train":
            print("warning: Integrated DEA is running without --init-from-baseline.")
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "Integrated DEA and DEA-lite losses must not be enabled together."
            )
        if int(args.integrated_route_channels) < 1:
            raise ValueError("--integrated-route-channels must be >= 1.")
        if float(args.integrated_route_temperature) <= 0.0:
            raise ValueError("--integrated-route-temperature must be > 0.")
        if float(args.integrated_update_limit) <= 0.0:
            raise ValueError("--integrated-update-limit must be > 0.")
        if float(args.integrated_route_loss_weight) < 0.0:
            raise ValueError("--integrated-route-loss-weight must be non-negative.")
        if int(args.integrated_route_ramp_epochs) < 0:
            raise ValueError("--integrated-route-ramp-epochs must be non-negative.")
        if (
            args.integrated_routing_mode == "attention"
            and float(args.integrated_route_loss_weight) > 0.0
        ):
            raise ValueError(
                "Residual action supervision is not defined for the attention "
                "control; set --integrated-route-loss-weight 0."
            )
        if float(args.integrated_uncertain_margin) <= 0.1:
            raise ValueError(
                "--integrated-uncertain-margin must be > 0.1 to guarantee "
                "the initial all-uncertain route."
            )
        if (
            args.integrated_routing_mode == "dea"
            and
            args.integrated_scale_routing
            and args.integrated_route_upsample_mode not in ("nearest", "nearest-exact")
        ):
            raise ValueError(
                "Hard scale routing requires nearest/nearest-exact upsampling; "
                "continuous interpolation destroys target/clutter exclusivity."
            )
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)

    if args.model_type == "full_dea":
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError("--if-checkpoint and --init-from-baseline are separate paths.")
        if not args.init_from_baseline and not args.if_checkpoint:
            print("warning: Full DEA is running without --init-from-baseline.")
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "Full DEA and DEA-lite losses must not be enabled together."
            )
        if args.full_dea_safe_kernel <= 0 or args.full_dea_safe_kernel % 2 == 0:
            raise ValueError("--full-dea-safe-kernel must be a positive odd integer.")
        for name in (
            "full_dea_topk_ratio",
            "full_dea_max_hard_bg_ratio",
        ):
            value = float(getattr(args, name))
            if value < 0.0 or value > 1.0:
                raise ValueError("--%s must be in [0, 1]." % name.replace("_", "-"))
        if float(args.full_dea_topk_min_score) < 0.0:
            raise ValueError("--full-dea-topk-min-score must be non-negative.")
        if args.full_dea_version not in ("v2", "v3", "v4", "v5"):
            raise ValueError("--full-dea-version must be v2, v3, v4, or v5.")
        if args.full_dea_protect_kernel <= 0 or args.full_dea_protect_kernel % 2 == 0:
            raise ValueError("--full-dea-protect-kernel must be a positive odd integer.")
        if args.full_dea_hard_min_area < 1:
            raise ValueError("--full-dea-hard-min-area must be >= 1.")
        if (
            args.full_dea_hard_max_area > 0
            and args.full_dea_hard_max_area < args.full_dea_hard_min_area
        ):
            raise ValueError(
                "--full-dea-hard-max-area must be 0 or >= --full-dea-hard-min-area."
            )
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)

    if is_cev_control(args.model_type):
        if args.if_checkpoint and args.init_from_baseline:
            raise ValueError(
                "--if-checkpoint and --init-from-baseline are separate paths."
            )
        if (
            getattr(args, "mode", "train") == "train"
            and not args.if_checkpoint
            and not args.init_from_baseline
        ):
            raise ValueError(
                "CEV controls must start from a complete trained MSHNet; "
                "pass --init-from-baseline."
            )
        kernel_size = int(getattr(args, "cev_kernel_size", 7))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("--cev-kernel-size must be a positive odd integer.")
        veto_strength = float(getattr(args, "cev_veto_strength", 1.0))
        if not 0.0 <= veto_strength <= 1.0:
            raise ValueError("--cev-veto-strength must be in [0,1].")
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "CEV controls and DEA-lite losses must not be enabled together."
            )
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)

    if is_dea_main_model(args.model_type):
        if int(args.predictive_state_channels) < 4:
            raise ValueError("--predictive-state-channels must be >= 4.")
        if not 0.0 < float(args.predictive_step_size) <= 1.0:
            raise ValueError("--predictive-step-size must be in (0, 1].")
        if float(args.predictive_delta_min) <= 0.0:
            raise ValueError("--predictive-delta-min must be > 0.")
        if float(args.predictive_delta_init) <= float(args.predictive_delta_min):
            raise ValueError(
                "--predictive-delta-init must be greater than "
                "--predictive-delta-min."
            )
        if int(args.predictive_log_interval) < 0:
            raise ValueError("--predictive-log-interval must be non-negative.")
        if args.init_from_baseline and not osp.isfile(args.init_from_baseline):
            raise FileNotFoundError(args.init_from_baseline)
        lite_lambdas = (
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        )
        if any(float(value) != 0.0 for value in lite_lambdas):
            raise ValueError(
                "DEA main-model training and DEA-lite losses must not be mixed."
            )

    if getattr(args, "mode", "train") == "train" and not getattr(args, "val_split_file", ""):
        val_fraction = float(getattr(args, "val_fraction", 0.2))
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("--val-fraction must be strictly between 0 and 1.")
    return args

def get_method_name(args):
    deep_supervision = getattr(args, "deep_supervision", "legacy_exact")
    if deep_supervision == "legacy":
        deep_supervision = "legacy_exact"
    if args.model_type == "mshnet" and deep_supervision != "legacy_exact":
        names = {
            "legacy_rescaled": "MSHNet-LegacyRescaled",
            "final_only": "MSHNet-FinalOnly",
            "side_no_location": "MSHNet-SideNoLocation",
            "rods_interval": "RODS-Interval",
            "rods_hard": "RODS-Hard",
            "rods_random": "RODS-Random",
            "rods_area_only": "RODS-AreaOnly",
            "tfds_projection": "TCDS-Projection",
            "tfds_projection_active_renorm": "TCDS-Projection-ActiveRenorm",
            "tgds_halfspace": "TGDS-Halfspace",
            "tsds_lift": "TSDS-Lift",
            "prds_regret": "PRDS-Regret",
            "cscs_leave_one_out": "CSCS-LeaveOneOut",
            "sfds_filtration": "SFDS-Filtration",
            "asfs_anchor_filtration": "ASFS-AnchorFiltration",
            "rdfs_continuation": "RDFS-Continuation",
            "crs_flip_suppression": "SDRR-ScaleDeletionResponsibility",
            "crs_matched_random": "SDRR-ScaleBudgetRandomControl-Unmatched",
            "crs_same_pixel_random_scale": "SDRR-SamePixelRandomScaleControl",
            "crs_magnitude_nonpivotal": "SDRR-MagnitudeMatchedNonPivotalControl",
            "legacy_no_s3": "MSHNet-Control-NoS3",
            "legacy_no_s2s3": "MSHNet-Control-NoS2S3",
            "legacy_s0_only": "MSHNet-Control-S0Only",
            "legacy_s3_delayed": "MSHNet-Control-S3Delayed",
            "hms_continuation": "MSHNet-HMS-Continuation",
            "mcsls_null_safe": "MSHNet-MC-SLS",
            "zmsls_null_abstain": "MSHNet-ZM-SLS-Abstain",
        }
        if deep_supervision == "crs_flip_suppression":
            normalization = getattr(args, "sdrr_normalization", "event")
            if normalization != "event":
                return "SDRR-NormalizationControl-" + normalization
        return names.get(deep_supervision, "MSHNet-" + deep_supervision)
    if is_cev_control(args.model_type):
        kernel = int(getattr(args, "cev_kernel_size", 7))
        if args.model_type == "dea_fine_veto_control":
            return "FineScaleVeto-Control-K%d" % kernel
        return "SharedCEV-Control-K%d" % kernel
    if is_dea_main_model(args.model_type):
        eta = ("%g" % float(
            getattr(args, "predictive_step_size", 1.0)
        )).replace("-", "m").replace(".", "p")
        prefix = "DEA-v0" if args.model_type == "dea" else "PredictiveCorrection"
        name = "%s-C%d-Eta%s" % (
            prefix,
            int(getattr(args, "predictive_state_channels", 32)),
            eta,
        )
        if bool(getattr(args, "predictive_legacy_numerics", False)):
            name += "-LegacyNum"
        return name
    if args.model_type == "dea_integrated":
        routing_mode = getattr(args, "integrated_routing_mode", "dea")
        decoder_routing = bool(getattr(args, "integrated_decoder_routing", True))
        scale_routing = bool(getattr(args, "integrated_scale_routing", True))
        residual_aligned = (
            float(getattr(args, "integrated_route_loss_weight", 0.0)) > 0.0
        )
        suffix = "-ResidualAligned" if residual_aligned else "-UnsupervisedRoute"
        if routing_mode == "soft_tri":
            return "DEAIntegrated-SoftTri" + suffix
        if routing_mode == "attention":
            return "DEAIntegrated-Attention"
        if decoder_routing and scale_routing:
            return "DEAIntegrated" + suffix
        if decoder_routing:
            return "DEAIntegrated-DecoderOnly" + suffix
        if scale_routing:
            return "DEAIntegrated-ScaleOnly" + suffix
        return "DEAIntegrated-Identity" + suffix
    if args.model_type == "full_dea":
        version = getattr(args, "full_dea_version", "v3")
        if version == "v2":
            return "FullDEA-v2"
        if version == "v4":
            return "FullDEA-v4-CRR"
        if version == "v5":
            return "FullDEA-v5-CRR-HT"
        return "FullDEA-v3-TPS"
    if (
        args.dea_lambda_single > 0
        or args.dea_lambda_dec > 0
        or args.dea_lambda_empty > 0
    ):
        return "DEA-lite"
    if getattr(args, "init_from_baseline", ""):
        return "MSHNet-Continued"
    mshnet_variant = getattr(args, "mshnet_variant", "workbench")
    if args.model_type == "mshnet" and mshnet_variant == "official":
        return "MSHNet-OfficialForward"
    if args.model_type == "mshnet" and mshnet_variant == "deterministic":
        return "MSHNet-Deterministic"
    return "MSHNet"

def get_run_folder_name(args, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    safe_method = get_method_name(args).replace('/', '_')
    return '%s-%s' % (safe_method, timestamp)

def get_method_metadata(args):
    return {
        "method": get_method_name(args),
        "model_type": args.model_type,
        "mshnet_variant": getattr(args, "mshnet_variant", "workbench"),
        "cev_kernel_size": int(getattr(args, "cev_kernel_size", 7)),
        "cev_initial_bias": float(getattr(args, "cev_initial_bias", -6.0)),
        "cev_veto_strength": float(getattr(args, "cev_veto_strength", 1.0)),
        "cev_baseline_frozen": bool(is_cev_control(args.model_type)),
        "full_dea_version": getattr(args, "full_dea_version", ""),
        "init_from_baseline": getattr(
            args, "origin_baseline_checkpoint", args.init_from_baseline
        ),
        "full_dea_lambda": float(args.full_dea_lambda),
        "full_dea_ramp_epochs": int(args.full_dea_ramp_epochs),
        "full_dea_start_epoch": int(args.full_dea_start_epoch),
        "full_dea_freeze_backbone_epochs": int(args.full_dea_freeze_backbone_epochs),
        "full_dea_tau_base": float(args.full_dea_tau_base),
        "full_dea_tau_target": float(args.full_dea_tau_target),
        "full_dea_tau_scale": float(args.full_dea_tau_scale),
        "full_dea_topk_ratio": float(args.full_dea_topk_ratio),
        "full_dea_topk_min_score": float(args.full_dea_topk_min_score),
        "full_dea_max_hard_bg_ratio": float(args.full_dea_max_hard_bg_ratio),
        "full_dea_safe_kernel": int(args.full_dea_safe_kernel),
        "full_dea_protect_kernel": int(args.full_dea_protect_kernel),
        "full_dea_hard_min_area": int(args.full_dea_hard_min_area),
        "full_dea_hard_max_area": int(args.full_dea_hard_max_area),
        "dea_lambda_single": float(args.dea_lambda_single),
        "dea_lambda_dec": float(args.dea_lambda_dec),
        "dea_lambda_empty": float(args.dea_lambda_empty),
        "integrated_route_channels": int(getattr(args, "integrated_route_channels", 16)),
        "integrated_route_temperature": float(getattr(args, "integrated_route_temperature", 1.0)),
        "integrated_routing_mode": getattr(args, "integrated_routing_mode", ""),
        "integrated_decoder_routing": bool(getattr(args, "integrated_decoder_routing", False)),
        "integrated_scale_routing": bool(getattr(args, "integrated_scale_routing", False)),
        "integrated_route_upsample_mode": getattr(args, "integrated_route_upsample_mode", ""),
        "integrated_update_limit": float(getattr(args, "integrated_update_limit", 0.25)),
        "integrated_uncertain_margin": float(getattr(args, "integrated_uncertain_margin", 1.0)),
        "integrated_route_loss_weight": float(getattr(args, "integrated_route_loss_weight", 0.0)),
        "integrated_route_ramp_epochs": int(getattr(args, "integrated_route_ramp_epochs", 0)),
        "integrated_isolate_route_gradients": bool(
            getattr(args, "integrated_isolate_route_gradients", True)
        ),
        "deep_supervision": getattr(args, "deep_supervision", "legacy_exact"),
        "aux_loss_weight": float(getattr(args, "aux_loss_weight", 0.8)),
        "ownership_preferred_cells": float(
            getattr(args, "ownership_preferred_cells", 3.0)
        ),
        "ownership_sigma": float(getattr(args, "ownership_sigma", 0.75)),
        "ownership_min_decidability": float(
            getattr(args, "ownership_min_decidability", 0.25)
        ),
        "ownership_interval_ratio": float(
            getattr(args, "ownership_interval_ratio", 0.5)
        ),
        "ownership_fallback": getattr(args, "ownership_fallback", "side0"),
        "ownership_ignore_dilation": int(
            getattr(args, "ownership_ignore_dilation", 3)
        ),
        "empty_side_policy": getattr(args, "empty_side_policy", "skip"),
        "rods_log_interval": int(getattr(args, "rods_log_interval", 50)),
        "tfds_min_iou": float(getattr(args, "tfds_min_iou", 0.5)),
        "tfds_max_centroid_distance": float(
            getattr(args, "tfds_max_centroid_distance", 3.0)
        ),
        "s3_start_epoch": int(getattr(args, "s3_start_epoch", 20)),
        "hms_ramp_epochs": int(getattr(args, "hms_ramp_epochs", 20)),
        "rdfs_start_epoch": int(getattr(args, "rdfs_start_epoch", 20)),
        "rdfs_ramp_epochs": int(getattr(args, "rdfs_ramp_epochs", 20)),
        "crs_lambda": float(getattr(args, "crs_lambda", 0.05)),
        "crs_start_epoch": int(getattr(args, "crs_start_epoch", 250)),
        "crs_ramp_epochs": int(getattr(args, "crs_ramp_epochs", 50)),
        "crs_safe_kernel": int(getattr(args, "crs_safe_kernel", 15)),
        "crs_detach_scale_evidence": bool(
            getattr(args, "crs_detach_scale_evidence", False)
        ),
        "sdrr_normalization": getattr(args, "sdrr_normalization", "event"),
        "predictive_state_channels": int(
            getattr(args, "predictive_state_channels", 32)
        ),
        "predictive_step_size": float(
            getattr(args, "predictive_step_size", 1.0)
        ),
        "predictive_delta_init": float(
            getattr(args, "predictive_delta_init", 1.0)
        ),
        "predictive_delta_min": float(
            getattr(args, "predictive_delta_min", 0.05)
        ),
        "predictive_legacy_numerics": bool(
            getattr(args, "predictive_legacy_numerics", False)
        ),
        "dea_version": "v0_adjoint_predictive_correction",
        "dea_state_channels": int(
            getattr(args, "predictive_state_channels", 32)
        ),
        "dea_step_size": float(
            getattr(args, "predictive_step_size", 1.0)
        ),
        "dea_delta_init": float(
            getattr(args, "predictive_delta_init", 1.0)
        ),
        "dea_delta_min": float(
            getattr(args, "predictive_delta_min", 0.05)
        ),
        "dea_legacy_numerics": bool(
            getattr(args, "predictive_legacy_numerics", False)
        ),
        "dataset_dir": args.dataset_dir,
        "train_split_file": getattr(args, "train_split_file", ""),
        "val_split_file": getattr(args, "val_split_file", ""),
        "test_split_file": getattr(args, "test_split_file", ""),
        "val_fraction": float(getattr(args, "val_fraction", 0.2)),
        "split_seed": int(getattr(args, "split_seed", getattr(args, "seed", 0))),
        "train_split_sha256": getattr(args, "train_split_sha256", ""),
        "val_split_sha256": getattr(args, "val_split_sha256", ""),
        "test_split_sha256": getattr(args, "test_split_sha256", ""),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "run_label": getattr(args, "run_label", ""),
    }

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--train-split-file', type=str, default='')
    parser.add_argument('--val-split-file', type=str, default='')
    parser.add_argument('--test-split-file', type=str, default='')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=20260706)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gpu-ids', type=str, default='')
    parser.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--if-checkpoint', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--reset-optimizer', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--weight-path', type=str, default=osp.join(DEFAULT_WEIGHT_DIR, 'IRSTD-1k_weight.tar'))
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument(
        '--run-dir',
        type=str,
        default='',
        help=(
            'Exact output directory for a new training run. Relative paths '
            'are resolved from the project root. The directory must be empty; '
            'use --if-checkpoint with --checkpoint-dir to resume.'
        ),
    )
    parser.add_argument(
        '--run-label',
        type=str,
        default='',
        help='Stable experiment label persisted in checkpoint metadata.',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='mshnet',
        choices=[
            'mshnet',
            'dea',
            'full_dea',
            'dea_integrated',
            'predictive_correction',
            'dea_fine_veto_control',
            'dea_cev_control',
        ],
    )
    parser.add_argument(
        '--mshnet-variant',
        choices=['workbench', 'official', 'deterministic'],
        default='workbench',
        help=(
            'Physical MSHNet implementation: historical experiment workbench, '
            'canonical official-forward, or parameter-identical deterministic-backward.'
        ),
    )
    parser.add_argument(
        '--deep-supervision',
        type=str,
        default='legacy_exact',
        choices=[
            'legacy',
            'legacy_exact',
            'legacy_rescaled',
            'final_only',
            'side_no_location',
            'rods_interval',
            'rods_hard',
            'rods_random',
            'rods_area_only',
            'tfds_projection',
            'tfds_projection_active_renorm',
            'tgds_halfspace',
            'tsds_lift',
            'prds_regret',
            'cscs_leave_one_out',
            'sfds_filtration',
            'asfs_anchor_filtration',
            'rdfs_continuation',
            'crs_flip_suppression',
            'crs_matched_random',
            'crs_same_pixel_random_scale',
            'crs_magnitude_nonpivotal',
            'legacy_no_s3',
            'legacy_no_s2s3',
            'legacy_s0_only',
            'legacy_s3_delayed',
            'hms_continuation',
            'mcsls_null_safe',
            'zmsls_null_abstain',
        ],
        help='Deep-supervision training topology for MSHNet.',
    )
    parser.add_argument('--aux-loss-weight', type=float, default=0.8)
    parser.add_argument('--ownership-preferred-cells', type=float, default=3.0)
    parser.add_argument('--ownership-sigma', type=float, default=0.75)
    parser.add_argument('--ownership-min-decidability', type=float, default=0.25)
    parser.add_argument('--ownership-interval-ratio', type=float, default=0.5)
    parser.add_argument(
        '--ownership-fallback',
        type=str,
        default='side0',
        choices=['side0', 'final_only'],
    )
    parser.add_argument('--ownership-ignore-dilation', type=int, default=3)
    parser.add_argument(
        '--empty-side-policy',
        type=str,
        default='skip',
        choices=['skip', 'background_only'],
    )
    parser.add_argument('--rods-log-interval', type=int, default=50)
    parser.add_argument('--tfds-min-iou', type=float, default=0.5)
    parser.add_argument('--tfds-max-centroid-distance', type=float, default=3.0)
    parser.add_argument('--s3-start-epoch', type=int, default=20)
    parser.add_argument('--hms-ramp-epochs', type=int, default=20)
    parser.add_argument('--rdfs-start-epoch', type=int, default=20)
    parser.add_argument('--rdfs-ramp-epochs', type=int, default=20)
    parser.add_argument('--crs-lambda', type=float, default=0.05)
    parser.add_argument('--crs-start-epoch', type=int, default=250)
    parser.add_argument('--crs-ramp-epochs', type=int, default=50)
    parser.add_argument('--crs-safe-kernel', type=int, default=15)
    parser.add_argument(
        '--sdrr-normalization',
        choices=['event', 'safe_density', 'unique_pixel'],
        default='event',
        help='Attribution control for SDRR gradient-budget normalization.',
    )
    parser.add_argument(
        '--crs-detach-scale-evidence', type=str2bool, default=False,
        help=(
            'Backpropagate the responsibility term only into the native final '
            'fusion convolution; canonical segmentation gradients still train '
            'the full MSHNet.'
        ),
    )
    parser.add_argument('--init-from-baseline', type=str, default='')
    parser.add_argument('--cev-kernel-size', type=int, default=7)
    parser.add_argument('--cev-initial-bias', type=float, default=-6.0)
    parser.add_argument('--cev-veto-strength', type=float, default=1.0)
    parser.add_argument('--dea-lambda-single', type=float, default=0.0)
    parser.add_argument('--dea-lambda-dec', type=float, default=0.0)
    parser.add_argument('--dea-lambda-empty', type=float, default=0.0)
    parser.add_argument('--dea-tau', type=float, default=0.5)
    parser.add_argument('--dea-ramp-epochs', type=int, default=0)
    parser.add_argument('--save-dea-debug', action='store_true')
    parser.add_argument('--dea-debug-interval', type=int, default=50)
    parser.add_argument('--dea-debug-max-batches', type=int, default=1)
    parser.add_argument('--dea-detach-evidence', action='store_true')
    parser.add_argument('--seed', type=int, default=20260706)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pd-fa-min-pd', type=float, default=0.93)
    parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
    parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
    parser.add_argument('--pd-fa-iou-margin', type=float, default=0.005)
    parser.add_argument('--full-dea-lambda', type=float, default=1.0)
    parser.add_argument(
        '--full-dea-version',
        type=str,
        default='v3',
        choices=['v2', 'v3', 'v4', 'v5'],
    )
    parser.add_argument('--full-dea-ramp-epochs', type=int, default=30)
    parser.add_argument('--full-dea-start-epoch', type=int, default=0)
    parser.add_argument('--full-dea-freeze-backbone-epochs', type=int, default=0)
    parser.add_argument('--full-dea-tau-base', type=float, default=0.45)
    parser.add_argument('--full-dea-tau-target', type=float, default=0.45)
    parser.add_argument('--full-dea-tau-scale', type=float, default=0.45)
    parser.add_argument('--full-dea-topk-ratio', type=float, default=0.001)
    parser.add_argument('--full-dea-topk-min-score', type=float, default=0.45)
    parser.add_argument('--full-dea-max-hard-bg-ratio', type=float, default=0.003)
    parser.add_argument('--full-dea-safe-kernel', type=int, default=15)
    parser.add_argument('--full-dea-protect-kernel', type=int, default=9)
    parser.add_argument('--full-dea-hard-min-area', type=int, default=1)
    parser.add_argument('--full-dea-hard-max-area', type=int, default=256)
    parser.add_argument('--full-dea-debug', action='store_true')
    parser.add_argument('--integrated-route-channels', type=int, default=16)
    parser.add_argument('--integrated-route-temperature', type=float, default=1.0)
    parser.add_argument(
        '--integrated-routing-mode',
        type=str,
        default='dea',
        choices=['dea', 'soft_tri', 'attention'],
    )
    parser.add_argument(
        '--integrated-decoder-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        '--integrated-scale-routing',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        '--integrated-route-upsample-mode',
        type=str,
        default='nearest-exact',
        choices=['nearest', 'nearest-exact', 'bilinear', 'bicubic'],
    )
    parser.add_argument('--integrated-update-limit', type=float, default=0.25)
    parser.add_argument('--integrated-uncertain-margin', type=float, default=1.0)
    # Experimental identifiability control.  It is disabled by default because
    # the first real-data smoke drove every hard route to keep/abstain; it must
    # not be presented as a validated formal objective without the prescribed
    # controls in the experiment protocol.
    parser.add_argument('--integrated-route-loss-weight', type=float, default=0.0)
    parser.add_argument('--integrated-route-ramp-epochs', type=int, default=3)
    parser.add_argument(
        '--dea-state-channels', '--predictive-state-channels',
        dest='predictive_state_channels', type=int, default=32,
    )
    parser.add_argument(
        '--dea-step-size', '--predictive-step-size',
        dest='predictive_step_size', type=float, default=1.0,
    )
    parser.add_argument(
        '--dea-delta-init', '--predictive-delta-init',
        dest='predictive_delta_init', type=float, default=1.0,
    )
    parser.add_argument(
        '--dea-delta-min', '--predictive-delta-min',
        dest='predictive_delta_min', type=float, default=0.05,
    )
    parser.add_argument(
        '--dea-legacy-numerics', '--predictive-legacy-numerics',
        dest='predictive_legacy_numerics',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument('--predictive-log-interval', type=int, default=50)
    parser.add_argument(
        '--integrated-isolate-route-gradients',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument('--integrated-log-interval', type=int, default=50)

    args = parser.parse_args()
    return validate_args(args)

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        setattr(args, 'origin_baseline_checkpoint', args.init_from_baseline)
        self.start_epoch = 0   
        self.mode = args.mode

        self.train_dataset = None
        if args.mode == 'train':
            trainset = IRSTD_Dataset(args, mode='train')
            valset = IRSTD_Dataset(args, mode='val')
            test_reference = IRSTD_Dataset(args, mode='test')
            self.assert_disjoint_splits(trainset, valset, test_reference)
            self.train_dataset = trainset
            setattr(args, 'train_split_sha256', trainset.split_sha256)
            setattr(args, 'val_split_sha256', valset.split_sha256)
            setattr(args, 'test_split_sha256', test_reference.split_sha256)
        else:
            valset = IRSTD_Dataset(args, mode='test')
            setattr(args, 'test_split_sha256', valset.split_sha256)
        self.val_dataset = valset

        def loader_kwargs(generator_seed):
            data_generator = torch.Generator()
            data_generator.manual_seed(generator_seed)
            kwargs = {
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.num_workers > 0,
                "worker_init_fn": seed_worker,
                "generator": data_generator,
            }
            if args.num_workers > 0:
                kwargs["prefetch_factor"] = 2
            return kwargs

        self.train_loader = None
        if self.train_dataset is not None:
            self.train_loader = Data.DataLoader(
                self.train_dataset,
                args.batch_size,
                shuffle=True,
                drop_last=True,
                **loader_kwargs(args.seed),
            )
        self.val_loader = Data.DataLoader(
            valset,
            1,
            drop_last=False,
            **loader_kwargs(args.seed + 1),
        )
        self.print_split_summary()

        device = torch.device('cuda')
        self.device = device
        torch.backends.cudnn.benchmark = not args.deterministic

        if args.model_type == "dea_fine_veto_control":
            model = FineScaleVetoMSHNet(
                3,
                kernel_size=args.cev_kernel_size,
                initial_bias=args.cev_initial_bias,
                veto_strength=args.cev_veto_strength,
                freeze_baseline=True,
            )
        elif args.model_type == "dea_cev_control":
            model = SharedCEVMSHNet(
                3,
                kernel_size=args.cev_kernel_size,
                initial_bias=args.cev_initial_bias,
                veto_strength=args.cev_veto_strength,
                freeze_baseline=True,
            )
        elif args.model_type == "full_dea":
            model = FullDEAMSHNet(3, full_dea_version=args.full_dea_version)
        elif args.model_type == "dea_integrated":
            model = DEAIntegratedMSHNet(
                3,
                route_channels=args.integrated_route_channels,
                route_temperature=args.integrated_route_temperature,
                routing_mode=args.integrated_routing_mode,
                decoder_routing=args.integrated_decoder_routing,
                scale_routing=args.integrated_scale_routing,
                route_upsample_mode=args.integrated_route_upsample_mode,
                update_limit=args.integrated_update_limit,
                uncertain_margin=args.integrated_uncertain_margin,
                isolate_route_gradients=args.integrated_isolate_route_gradients,
            )
        elif is_dea_main_model(args.model_type):
            model = DEAMSHNet(
                3,
                state_channels=args.predictive_state_channels,
                step_size=args.predictive_step_size,
                delta_init=args.predictive_delta_init,
                delta_min=args.predictive_delta_min,
                legacy_influence_numerics=args.predictive_legacy_numerics,
            )
        elif args.model_type == "mshnet" and args.mshnet_variant == "official":
            model = OfficialBaselineMSHNet(3)
        elif args.model_type == "mshnet" and args.mshnet_variant == "deterministic":
            model = DeterministicBaselineMSHNet(3)
        else:
            model = MSHNet(3)

        if args.multi_gpus and torch.cuda.device_count() > 1:
            device_ids = self.parse_gpu_ids(args.gpu_ids)
            print('use %d gpus: %s' % (len(device_ids), device_ids))
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        self.model = model

        if args.mode == 'train' and args.init_from_baseline and not args.if_checkpoint:
            baseline = load_torch_file(args.init_from_baseline)
            state_dict = self.extract_state_dict(baseline)
            if is_cev_control(args.model_type):
                allowed_missing = CounterfactualVetoMSHNet.BASELINE_MISSING_PREFIXES
                allowed_unexpected = (
                    CounterfactualVetoMSHNet.BASELINE_UNEXPECTED_PREFIXES
                )
            elif args.model_type == "dea_integrated":
                allowed_missing = DEAIntegratedMSHNet.BASELINE_MISSING_PREFIXES
                allowed_unexpected = DEAIntegratedMSHNet.BASELINE_UNEXPECTED_PREFIXES
            elif is_dea_main_model(args.model_type):
                allowed_missing = DEAMSHNet.BASELINE_MISSING_PREFIXES
                allowed_unexpected = DEAMSHNet.BASELINE_UNEXPECTED_PREFIXES
            elif args.model_type == "full_dea":
                allowed_missing = ("full_dea_head.", "decidability_head.")
                allowed_unexpected = ()
            else:
                # The pristine public MSHNet checkpoint predates the optional
                # DEA-lite head present in this repository.
                allowed_missing = ("decidability_head.",)
                allowed_unexpected = ()
            self.load_model_state_partial(
                state_dict,
                allowed_missing_prefixes=allowed_missing,
                allowed_unexpected_prefixes=allowed_unexpected,
            )

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.masked_owned_loss = MaskedOwnedScaleIoULoss()
        self.partial_sls_loss = PartialSLSIoULoss()
        self.measure_conditioned_sls_loss = MeasureConditionedSLSIoULoss()
        self.zero_measure_sls_loss = MeasureConditionedSLSIoULoss(
            null_mode="abstain"
        )
        self.last_deep_supervision_log = {}
        self.last_tgds_components = None
        graph_mode = {
            "rods_interval": "interval",
            "rods_hard": "hard",
            "rods_random": "random",
            "rods_area_only": "area_only",
        }.get(getattr(args, "deep_supervision", "legacy_exact"), "interval")
        self.resolution_graph = ResolutionDecidableSupervisionGraph(
            preferred_diameter_cells=args.ownership_preferred_cells,
            sigma=args.ownership_sigma,
            min_decidability=args.ownership_min_decidability,
            interval_ratio=args.ownership_interval_ratio,
            mode=graph_mode,
            fallback=args.ownership_fallback,
        )
        self.owned_supervision_builder = OwnedSideSupervisionBuilder(
            ignore_dilation=args.ownership_ignore_dilation,
        )
        self.task_consistent_graph = TaskConsistentProjectionGraph(
            min_iou=args.tfds_min_iou,
            max_centroid_distance=args.tfds_max_centroid_distance,
        )
        self.task_consistent_target_builder = TaskConsistentPartialTargetBuilder()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0.0
        self.best_pd_fa = float('inf')
        self.best_pd_fa_iou = 0.0
        self.best_pd_fa_pd = 0.0
        self.best_pd_fa_epoch = -1
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = args.checkpoint_dir or self.find_latest_checkpoint_folder()
                checkpoint = load_torch_file(osp.join(check_folder, 'checkpoint.pkl'))
                self.validate_integrated_checkpoint_metadata(
                    checkpoint,
                    check_split_hashes=True,
                )
                if args.model_type == 'dea_integrated':
                    setattr(
                        args,
                        'origin_baseline_checkpoint',
                        checkpoint.get('method_meta', {}).get(
                            'init_from_baseline', ''
                        ),
                    )
                self.load_model_state(checkpoint['net'])
                if args.reset_optimizer:
                    print('reset optimizer state')
                else:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except (ValueError, RuntimeError) as exc:
                        print('skip optimizer state: %s' % exc)
                self.set_optimizer_lr(args.lr)
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.best_iou = float(checkpoint.get('best_iou', checkpoint.get('iou', 0.0)))
                self.best_pd_fa = float(checkpoint.get('best_pd_fa', float('inf')))
                self.best_pd_fa_iou = float(checkpoint.get('best_pd_fa_iou', 0.0))
                self.best_pd_fa_pd = float(checkpoint.get('best_pd_fa_pd', 0.0))
                self.best_pd_fa_epoch = int(checkpoint.get('best_pd_fa_epoch', -1))
                self.save_folder = check_folder
            else:
                requested_run_dir = getattr(args, 'run_dir', '')
                if requested_run_dir:
                    self.save_folder = requested_run_dir
                    if not osp.isabs(self.save_folder):
                        self.save_folder = osp.join(PROJECT_DIR, self.save_folder)
                    self.save_folder = osp.normpath(self.save_folder)
                    if osp.isdir(self.save_folder) and os.listdir(self.save_folder):
                        raise FileExistsError(
                            'Refusing to overwrite non-empty --run-dir %s; '
                            'resume with --if-checkpoint true '
                            '--checkpoint-dir %s.'
                            % (self.save_folder, self.save_folder)
                        )
                else:
                    self.save_folder = osp.join(
                        DEFAULT_WEIGHT_DIR,
                        get_run_folder_name(args),
                    )
                os.makedirs(self.save_folder, exist_ok=True)
            self.persist_split_manifests()
        if args.mode=='test':
          
            weight = load_torch_file(args.weight_path)
            self.validate_integrated_checkpoint_metadata(
                weight,
                check_split_hashes=False,
            )
            state_dict = self.extract_state_dict(weight)
            self.load_model_state(state_dict)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1

    @staticmethod
    def assert_disjoint_splits(trainset, valset, testset):
        named_sets = {
            'train': set(trainset.names),
            'val': set(valset.names),
            'test': set(testset.names),
        }
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = sorted(named_sets[left].intersection(named_sets[right]))
            if overlap:
                raise RuntimeError(
                    '%s/%s split leakage detected (%d samples), e.g. %s'
                    % (left, right, len(overlap), overlap[:5])
                )

    def print_split_summary(self):
        if self.train_dataset is not None:
            print(
                'split train: n=%d sha256=%s source=%s'
                % (
                    len(self.train_dataset),
                    self.train_dataset.split_sha256[:12],
                    self.train_dataset.split_source,
                )
            )
            print(
                'split val:   n=%d sha256=%s source=%s'
                % (
                    len(self.val_dataset),
                    self.val_dataset.split_sha256[:12],
                    self.val_dataset.split_source,
                )
            )
        else:
            print(
                'split test:  n=%d sha256=%s source=%s'
                % (
                    len(self.val_dataset),
                    self.val_dataset.split_sha256[:12],
                    self.val_dataset.split_source,
                )
            )

    def persist_split_manifests(self):
        if self.mode != 'train':
            return
        for split_name, dataset in (
            ('train', self.train_dataset),
            ('val', self.val_dataset),
        ):
            manifest_path = osp.join(self.save_folder, 'split_%s.txt' % split_name)
            with open(manifest_path, 'w') as f:
                for name in dataset.names:
                    f.write(name + '\n')

        config_path = osp.join(self.save_folder, 'run_config.json')
        serializable_args = {}
        for key, value in sorted(vars(self.args).items()):
            if value is None or isinstance(value, (bool, int, float, str)):
                serializable_args[key] = value
            else:
                serializable_args[key] = repr(value)
        with open(config_path, 'w') as f:
            json.dump(
                {
                    'args': serializable_args,
                    'method_meta': get_method_metadata(self.args),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write('\n')

    def validate_integrated_checkpoint_metadata(
        self,
        checkpoint,
        check_split_hashes,
    ):
        if (
            self.args.model_type != 'dea_integrated'
            and not is_dea_main_model(self.args.model_type)
            and not is_cev_control(self.args.model_type)
            and not (
                self.args.model_type == 'mshnet'
                and (
                    getattr(self.args, "deep_supervision", "legacy_exact")
                    != "legacy_exact"
                    or getattr(self.args, "mshnet_variant", "workbench")
                    != "workbench"
                )
            )
        ):
            return
        if not isinstance(checkpoint, dict) or 'method_meta' not in checkpoint:
            raise RuntimeError(
                '%s resume/test requires a checkpoint containing '
                'method_meta; use checkpoint.pkl/checkpoint_best_iou.pkl rather '
                'than a raw weight.pkl file.'
                % self.args.model_type
            )

        metadata = checkpoint['method_meta']
        expected = get_method_metadata(self.args)
        if is_cev_control(self.args.model_type):
            semantic_keys = (
                'model_type',
                'cev_kernel_size',
                'cev_initial_bias',
                'cev_veto_strength',
                'cev_baseline_frozen',
                'test_split_sha256',
            )
        elif self.args.model_type == 'dea_integrated':
            semantic_keys = (
                'model_type',
                'integrated_route_channels',
                'integrated_route_temperature',
                'integrated_routing_mode',
                'integrated_decoder_routing',
                'integrated_scale_routing',
                'integrated_route_upsample_mode',
                'integrated_update_limit',
                'integrated_uncertain_margin',
                'integrated_route_loss_weight',
                'integrated_route_ramp_epochs',
                'integrated_isolate_route_gradients',
                'test_split_sha256',
            )
        elif self.args.model_type == 'predictive_correction':
            semantic_keys = (
                'model_type',
                'predictive_state_channels',
                'predictive_step_size',
                'predictive_delta_init',
                'predictive_delta_min',
                'predictive_legacy_numerics',
                'test_split_sha256',
            )
        elif is_dea_main_model(self.args.model_type):
            semantic_keys = (
                'model_type',
                'dea_version',
                'dea_state_channels',
                'dea_step_size',
                'dea_delta_init',
                'dea_delta_min',
                'dea_legacy_numerics',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and is_tfds_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'tfds_min_iou',
                'tfds_max_centroid_distance',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and is_scheduled_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                's3_start_epoch',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and is_homotopy_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'hms_ramp_epochs',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and is_responsibility_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'crs_lambda',
                'crs_start_epoch',
                'crs_ramp_epochs',
                'crs_safe_kernel',
                'crs_detach_scale_evidence',
                'sdrr_normalization',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and getattr(self.args, "deep_supervision", "")
            == "rdfs_continuation"
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'rdfs_start_epoch',
                'rdfs_ramp_epochs',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and is_measure_conditioned_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'test_split_sha256',
            )
        elif (
            self.args.model_type == 'mshnet'
            and (
                is_tgds_deep_supervision(
                    getattr(self.args, "deep_supervision", "")
                )
                or is_tsds_deep_supervision(
                    getattr(self.args, "deep_supervision", "")
                )
                or is_prds_deep_supervision(
                    getattr(self.args, "deep_supervision", "")
                )
                or is_coalition_deep_supervision(
                    getattr(self.args, "deep_supervision", "")
                )
                or is_scale_subset_deep_supervision(
                    getattr(self.args, "deep_supervision", "")
                )
            )
        ):
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'test_split_sha256',
            )
        elif self.args.model_type == 'mshnet':
            semantic_keys = (
                'model_type',
                'deep_supervision',
                'aux_loss_weight',
                'ownership_preferred_cells',
                'ownership_sigma',
                'ownership_min_decidability',
                'ownership_interval_ratio',
                'ownership_fallback',
                'ownership_ignore_dilation',
                'empty_side_policy',
                'test_split_sha256',
            )
        if (
            self.args.model_type == 'mshnet'
            and getattr(self.args, "mshnet_variant", "workbench") != "workbench"
        ):
            semantic_keys = ('mshnet_variant',) + semantic_keys
        if check_split_hashes:
            semantic_keys = semantic_keys + (
                'train_split_sha256',
                'val_split_sha256',
            )

        mismatches = []
        for key in semantic_keys:
            if key not in metadata:
                if (
                    key == 'sdrr_normalization'
                    and expected[key] == 'event'
                ):
                    # Backward-compatible semantics for formal checkpoints
                    # created before the normalization control was named.
                    continue
                mismatches.append('%s=<missing>' % key)
            elif metadata[key] != expected[key]:
                mismatches.append(
                    '%s checkpoint=%r cli=%r'
                    % (key, metadata[key], expected[key])
                )
        if mismatches:
            raise RuntimeError(
                '%s checkpoint semantics mismatch: %s'
                % (self.args.model_type, '; '.join(mismatches))
            )

    def parse_gpu_ids(self, gpu_ids):
        if gpu_ids:
            device_ids = [int(item) for item in gpu_ids.split(',') if item.strip()]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise ValueError('No GPU ids selected.')
        return device_ids

    def extract_state_dict(self, weight_obj):
        if isinstance(weight_obj, dict):
            if 'state_dict' in weight_obj:
                return weight_obj['state_dict']
            if 'net' in weight_obj:
                return weight_obj['net']

            looks_like_state_dict = all(
                torch.is_tensor(value) for value in weight_obj.values()
            )
            if looks_like_state_dict:
                return weight_obj

        raise RuntimeError(
            'Unsupported weight format. Expected raw state_dict, '
            'dict with state_dict, or dict with net.'
        )

    def load_model_state(self, state_dict):
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        if isinstance(self.model, nn.DataParallel):
            try:
                self.model.module.load_state_dict(state_dict)
                return
            except RuntimeError:
                pass

        has_module_prefix = all(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            target_model.load_state_dict(state_dict)
            return

        raise RuntimeError('Failed to load model state_dict.')

    def load_model_state_partial(
        self,
        state_dict,
        allowed_missing_prefixes=(),
        allowed_unexpected_prefixes=(),
    ):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

        missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
        bad_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        bad_unexpected = [
            key
            for key in unexpected
            if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                'Partial baseline load failed. bad_missing=%s bad_unexpected=%s'
                % (bad_missing, bad_unexpected)
            )
        print(
            'loaded baseline with partial state: missing=%d unexpected=%d'
            % (len(missing), len(unexpected))
        )

    def set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('set optimizer lr: %.6f' % lr)

    def find_latest_checkpoint_folder(self):
        run_glob = '%s-*' % get_method_name(self.args)
        checkpoint_paths = sorted(
            glob.glob(osp.join(DEFAULT_WEIGHT_DIR, run_glob, 'checkpoint.pkl')),
            key=osp.getmtime,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                'No %s checkpoint found under %s. Pass --checkpoint-dir inside the project weight directory.'
                % (get_method_name(self.args), DEFAULT_WEIGHT_DIR)
            )
        return osp.dirname(checkpoint_paths[-1])
        
    def use_dea(self, epoch):
        return (
            self.args.model_type == "mshnet"
            and
            epoch > self.warm_epoch
            and (
                self.args.dea_lambda_single > 0
                or self.args.dea_lambda_dec > 0
                or self.args.dea_lambda_empty > 0
            )
        )

    def get_forward_tag(self, epoch):
        if is_cev_control(self.args.model_type):
            # CEV is a frozen-checkpoint control over the complete four-scale
            # MSHNet graph; there is no cold single-head training phase.
            return True
        if self.args.model_type == "full_dea":
            return epoch >= self.args.full_dea_start_epoch
        if self.args.model_type == "dea_integrated":
            # Integrated DEA is a continued-training method over a complete
            # MSHNet checkpoint; terminal scale routing must be exercised from
            # the first optimization step.
            return True
        if is_homotopy_deep_supervision(
            getattr(self.args, "deep_supervision", "")
        ):
            return True
        return epoch > self.warm_epoch

    def get_hms_alpha(self, epoch):
        if not is_homotopy_deep_supervision(
            getattr(self.args, "deep_supervision", "")
        ):
            return 1.0
        return min(
            1.0,
            max(0.0, float(epoch) / float(self.args.hms_ramp_epochs)),
        )

    def get_rdfs_alpha(self, epoch):
        if getattr(self.args, "deep_supervision", "") != "rdfs_continuation":
            return 1.0
        start = int(self.args.rdfs_start_epoch)
        ramp = int(self.args.rdfs_ramp_epochs)
        return min(1.0, max(0.0, float(epoch - start) / float(ramp)))

    def get_full_dea_ramp(self, epoch):
        if epoch < self.args.full_dea_start_epoch:
            return 0.0
        return get_dea_ramp(
            epoch,
            self.args.full_dea_start_epoch - 1,
            self.args.full_dea_ramp_epochs,
        )

    def configure_full_dea_trainable(self, epoch):
        if self.args.model_type != "full_dea":
            return

        freeze = epoch < self.args.full_dea_freeze_backbone_epochs
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for name, param in model.named_parameters():
            if freeze:
                param.requires_grad = name.startswith("full_dea_head")
            else:
                param.requires_grad = True

        if freeze:
            for name, module in model.named_modules():
                if name.startswith("full_dea_head"):
                    continue
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            model.full_dea_head.train()

    def format_log_dict(self, log_dict):
        msg = []
        for key, value in log_dict.items():
            try:
                scalar = float(value.detach().mean()) if torch.is_tensor(value) else float(value)
                msg.append('%s=%.6f' % (key, scalar))
            except (TypeError, ValueError):
                pass
        return msg

    @staticmethod
    def new_integrated_route_audit():
        return {
            "route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "gt_route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "bg_route_counts": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "entropy_sum": [0.0] * 4,
            "entropy_count": [0] * 4,
            "transitions": [torch.zeros((3, 3), dtype=torch.long) for _ in range(3)],
            "delta_abs_sum": [0.0] * 4,
            "delta_count": [0] * 4,
            "target_prob_fn_sum": [0.0] * 4,
            "target_prob_fn_count": [0] * 4,
            "clutter_prob_fp_sum": [0.0] * 4,
            "clutter_prob_fp_count": [0] * 4,
            "keep_prob_correct_sum": [0.0] * 4,
            "keep_prob_correct_count": [0] * 4,
            "hard_action_condition_hits": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "hard_action_condition_totals": [torch.zeros(3, dtype=torch.long) for _ in range(4)],
            "delta_residual_aligned": [0] * 4,
            "delta_residual_active": [0] * 4,
        }

    @staticmethod
    def update_integrated_route_audit(audit, output, target):
        routes = output["routes"]
        preclosure_probability = torch.sigmoid(
            output["scale_fusion"]["z_base"].detach()
        )
        binary_target_full = target > 0.5
        false_negative = binary_target_full & (preclosure_probability < 0.5)
        false_positive = (~binary_target_full) & (preclosure_probability >= 0.5)
        correct = ~(false_negative | false_positive)
        for scale, route in enumerate(routes):
            winner = route["winner"].detach()
            probabilities = route["probabilities"].detach().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1)
            audit["route_counts"][scale] += torch.bincount(
                winner.reshape(-1).cpu(), minlength=3
            )
            audit["entropy_sum"][scale] += float(entropy.sum().item())
            audit["entropy_count"][scale] += int(entropy.numel())

            target_at_scale = F.adaptive_max_pool2d(
                target.float(), output_size=winner.shape[-2:]
            )[:, 0] > 0.5
            gt_winner = winner[target_at_scale]
            bg_winner = winner[~target_at_scale]
            if gt_winner.numel():
                audit["gt_route_counts"][scale] += torch.bincount(
                    gt_winner.reshape(-1).cpu(), minlength=3
                )
            if bg_winner.numel():
                audit["bg_route_counts"][scale] += torch.bincount(
                    bg_winner.reshape(-1).cpu(), minlength=3
                )

            probabilities_full = F.interpolate(
                probabilities,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            winner_full = F.interpolate(
                winner.unsqueeze(1).float(),
                size=target.shape[-2:],
                mode='nearest',
            ).long()
            conditions = (
                (false_negative, 0, "target_prob_fn"),
                (false_positive, 1, "clutter_prob_fp"),
                (correct, 2, "keep_prob_correct"),
            )
            for condition, action, prefix in conditions:
                count = int(condition.sum().item())
                audit[prefix + "_count"][scale] += count
                if count:
                    audit[prefix + "_sum"][scale] += float(
                        probabilities_full[:, action:action + 1][condition].sum().item()
                    )
                    audit["hard_action_condition_hits"][scale][action] += int(
                        (winner_full[condition] == action).sum().item()
                    )
                audit["hard_action_condition_totals"][scale][action] += count

        # routes are fine-to-coarse.  Each matrix row is the coarse state and
        # each column is the corresponding next-finer state.
        for fine_scale in range(3):
            fine = routes[fine_scale]["winner"].detach()
            coarse = routes[fine_scale + 1]["winner"].detach()
            coarse_up = F.interpolate(
                coarse.unsqueeze(1).float(),
                size=fine.shape[-2:],
                mode='nearest',
            ).squeeze(1).long()
            pair_index = (coarse_up * 3 + fine).reshape(-1).cpu()
            audit["transitions"][fine_scale] += torch.bincount(
                pair_index, minlength=9
            ).reshape(3, 3)

        deltas = output["scale_fusion"]["deltas"].detach()
        for scale in range(4):
            scale_delta = deltas[:, scale]
            audit["delta_abs_sum"][scale] += float(scale_delta.abs().sum().item())
            audit["delta_count"][scale] += int(scale_delta.numel())
            desired_direction = (
                binary_target_full.to(scale_delta.dtype)
                - preclosure_probability
            )[:, 0]
            active = scale_delta.abs() > 0
            audit["delta_residual_active"][scale] += int(active.sum().item())
            audit["delta_residual_aligned"][scale] += int(
                ((scale_delta * desired_direction > 0) & active).sum().item()
            )

    def finalize_integrated_route_audit(self, audit, epoch):
        def normalized(counts):
            total = int(counts.sum().item())
            if total == 0:
                return [0.0, 0.0, 0.0]
            return [float(value) / float(total) for value in counts.tolist()]

        report = {
            "epoch": int(epoch),
            "mode": self.mode,
            "evaluation_split_sha256": self.val_dataset.split_sha256,
            "state_order": [
                "increase/target",
                "decrease/clutter",
                "keep/abstain",
            ],
            "route_occupancy": [normalized(item) for item in audit["route_counts"]],
            "gt_route_occupancy": [normalized(item) for item in audit["gt_route_counts"]],
            "bg_route_occupancy": [normalized(item) for item in audit["bg_route_counts"]],
            "route_entropy": [
                audit["entropy_sum"][scale]
                / max(1, audit["entropy_count"][scale])
                for scale in range(4)
            ],
            "coarse_to_fine_transitions": [
                matrix.tolist() for matrix in audit["transitions"]
            ],
            "scale_delta_abs_mean": [
                audit["delta_abs_sum"][scale]
                / max(1, audit["delta_count"][scale])
                for scale in range(4)
            ],
            "target_probability_on_false_negative": [
                audit["target_prob_fn_sum"][scale]
                / max(1, audit["target_prob_fn_count"][scale])
                for scale in range(4)
            ],
            "clutter_probability_on_false_positive": [
                audit["clutter_prob_fp_sum"][scale]
                / max(1, audit["clutter_prob_fp_count"][scale])
                for scale in range(4)
            ],
            "keep_probability_on_correct": [
                audit["keep_prob_correct_sum"][scale]
                / max(1, audit["keep_prob_correct_count"][scale])
                for scale in range(4)
            ],
            "hard_action_condition_accuracy": [
                [
                    float(audit["hard_action_condition_hits"][scale][action])
                    / max(
                        1,
                        int(audit["hard_action_condition_totals"][scale][action]),
                    )
                    for action in range(3)
                ]
                for scale in range(4)
            ],
            "delta_residual_sign_alignment": [
                float(audit["delta_residual_aligned"][scale])
                / max(1, audit["delta_residual_active"][scale])
                for scale in range(4)
            ],
        }
        serialized = json.dumps(report, sort_keys=True)
        print('[INTEGRATED DEA VAL] ' + serialized)
        if self.mode == 'train':
            with open(osp.join(self.save_folder, 'route_metric.jsonl'), 'a') as f:
                f.write(serialized + '\n')

        if all(item[2] > 0.999999 for item in report["route_occupancy"]):
            print(
                'warning: Integrated DEA remains all keep/abstain on the entire '
                'evaluation split; the current forward mapping is still the baseline.'
            )
        return report

    def save_dea_debug(self, epoch, iteration, data, labels, pred, dea_out):
        if not self.args.save_dea_debug:
            return
        if iteration >= self.args.dea_debug_max_batches:
            return
        if self.args.dea_debug_interval > 0 and epoch % self.args.dea_debug_interval != 0:
            return

        debug_root = self.save_folder if self.save_folder else PROJECT_DIR
        debug_dir = osp.join(debug_root, 'dea_debug')
        os.makedirs(debug_dir, exist_ok=True)

        sample = {
            "image": data[:1].detach().cpu(),
            "label": labels[:1].detach().cpu(),
            "z_full": pred[:1].detach().cpu(),
            "p_full": torch.sigmoid(pred[:1]).detach().cpu(),
            "scale_logits": dea_out["scale_logits"][:1].detach().cpu(),
            "z_only": dea_out["z_only"][:1].detach().cpu(),
            "p_only": torch.sigmoid(dea_out["z_only"][:1]).detach().cpu(),
            "z_only_max": dea_out["z_only_max"][:1].detach().cpu(),
            "p_only_max": torch.sigmoid(dea_out["z_only_max"][:1]).detach().cpu(),
            "z_empty": dea_out["z_empty"][:1].detach().cpu(),
            "p_empty": torch.sigmoid(dea_out["z_empty"][:1]).detach().cpu(),
            "d_logit": dea_out["decidability_logit"][:1].detach().cpu(),
            "d_prob": torch.sigmoid(dea_out["decidability_logit"][:1]).detach().cpu(),
        }

        torch.save(sample, osp.join(debug_dir, 'epoch_%04d_iter_%04d.pt' % (epoch, iteration)))

    def dea_main_loss(self, state_logits, labels, epoch):
        """Match MSHNet's effective resolution weights without duplicate heads.

        MSHNet averages final, full-resolution side, half, quarter, and eighth
        losses.  The first two together give the full-resolution prediction a
        weight of 0.4.  The predictive decoder has only one shared readout, so
        it applies 0.4 once to the final state and 0.2 to the preceding three
        states.  The coarsest 1/16 state is kept as an unsupervised prefix.
        """
        if len(state_logits) != 5:
            raise ValueError(
                "predictive decoder must return five coarse-to-fine logits"
            )
        final_loss = self.loss_fun(
            state_logits[4], labels, self.warm_epoch, epoch
        )
        if epoch <= self.warm_epoch:
            return final_loss

        labels_half = self.down(labels)
        labels_quarter = self.down(labels_half)
        labels_eighth = self.down(labels_quarter)
        loss_half = self.loss_fun(
            state_logits[3], labels_half, self.warm_epoch, epoch
        )
        loss_quarter = self.loss_fun(
            state_logits[2], labels_quarter, self.warm_epoch, epoch
        )
        loss_eighth = self.loss_fun(
            state_logits[1], labels_eighth, self.warm_epoch, epoch
        )
        return (
            0.4 * final_loss
            + 0.2 * loss_half
            + 0.2 * loss_quarter
            + 0.2 * loss_eighth
        )

    @staticmethod
    def unpack_batch(batch):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
        raise RuntimeError("unexpected dataloader batch arity: %d" % len(batch))

    def compute_deep_supervision_loss(
        self,
        pred,
        masks,
        labels,
        instance_map,
        epoch,
    ):
        mode = getattr(self.args, "deep_supervision", "legacy_exact")
        if mode == "legacy":
            mode = "legacy_exact"

        final_loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)
        self.last_deep_supervision_log = {
            "final_loss_raw": final_loss.detach(),
        }
        self.last_tgds_components = None
        if is_measure_conditioned_deep_supervision(mode):
            conditioned_loss = (
                self.zero_measure_sls_loss
                if mode == "zmsls_null_abstain"
                else self.measure_conditioned_sls_loss
            )
            final_loss = conditioned_loss(
                pred,
                labels,
                self.warm_epoch,
                epoch,
            )
            self.last_deep_supervision_log["final_loss_raw"] = (
                final_loss.detach()
            )
            loss = final_loss
            labels_for_scale = labels
            null_ratios = [
                (labels.flatten(1).sum(dim=1) == 0).float().mean()
            ]
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                side_loss = conditioned_loss(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                loss = loss + side_loss
                null_ratio = (
                    labels_for_scale.flatten(1).sum(dim=1) == 0
                ).float().mean()
                null_ratios.append(null_ratio)
                self.last_deep_supervision_log.update({
                    "side%d_loss" % side_index: side_loss.detach(),
                    "side%d_null_ratio" % side_index: null_ratio,
                })
            self.last_deep_supervision_log.update({
                "final_null_ratio": null_ratios[0],
                "measure_conditioned": 1.0,
                "canonical_aggregation": 1.0,
            })
            return loss / (len(masks) + 1)
        if mode == "legacy_exact":
            loss = final_loss
            labels_for_scale = labels
            for j in range(len(masks)):
                if j > 0:
                    labels_for_scale = self.down(labels_for_scale)
                loss = loss + self.loss_fun(
                    masks[j], labels_for_scale, self.warm_epoch, epoch
                )
            return loss / (len(masks) + 1)

        if is_homotopy_deep_supervision(mode):
            alpha = self.get_hms_alpha(epoch)
            labels_for_scale = labels
            side_losses = []
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                side_loss = self.loss_fun(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                side_losses.append(side_loss)
                self.last_deep_supervision_log[
                    "side%d_loss" % side_index
                ] = side_loss.detach()
            canonical = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            total = (1.0 - alpha) * final_loss + alpha * canonical
            self.last_deep_supervision_log.update({
                "hms_alpha": alpha,
                "canonical_loss": canonical.detach(),
                "homotopy_continuation": 1.0,
            })
            return total

        if is_scale_subset_deep_supervision(mode):
            active_indices = SCALE_SUBSET_DEEP_SUPERVISION[mode]
            labels_for_scale = labels
            side_losses = []
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                if side_index not in active_indices:
                    continue
                side_loss = self.loss_fun(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                side_losses.append(side_loss)
                self.last_deep_supervision_log[
                    "side%d_loss" % side_index
                ] = side_loss.detach()
            if not side_losses:
                self.last_deep_supervision_log["active_side_count"] = 0.0
                self.last_deep_supervision_log["subset_control"] = 1.0
                return final_loss
            total = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            self.last_deep_supervision_log["active_side_count"] = float(
                len(active_indices)
            )
            self.last_deep_supervision_log["subset_control"] = 1.0
            return total

        if is_scheduled_deep_supervision(mode):
            labels_for_scale = labels
            side_losses = []
            active_indices = (0, 1, 2) if epoch < self.args.s3_start_epoch else (0, 1, 2, 3)
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                if side_index not in active_indices:
                    continue
                side_loss = self.loss_fun(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                side_losses.append(side_loss)
                self.last_deep_supervision_log[
                    "side%d_loss" % side_index
                ] = side_loss.detach()
            if not side_losses:
                return final_loss
            total = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            self.last_deep_supervision_log.update({
                "active_side_count": float(len(active_indices)),
                "s3_enabled": float(3 in active_indices),
                "s3_start_epoch": float(self.args.s3_start_epoch),
                "scheduled_control": 1.0,
            })
            return total

        if not masks or mode == "final_only":
            return final_loss

        if is_responsibility_deep_supervision(mode):
            canonical_sum = final_loss
            labels_for_scale = labels
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                side_loss = self.loss_fun(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                canonical_sum = canonical_sum + side_loss
                self.last_deep_supervision_log[
                    "side%d_loss" % side_index
                ] = side_loss.detach()
            canonical_total = canonical_sum / (len(masks) + 1)
            start = int(self.args.crs_start_epoch)
            ramp = min(
                1.0,
                max(
                    0.0,
                    float(epoch - start) / float(self.args.crs_ramp_epochs),
                ),
            )
            if ramp == 0.0 or float(self.args.crs_lambda) == 0.0:
                self.last_deep_supervision_log.update({
                    "canonical_loss": canonical_total.detach(),
                    "crs_ramp": ramp,
                    "crs_identity": 1.0,
                })
                return canonical_total

            base_model = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            coalition = leave_one_scale_out_coalitions(
                (
                    tuple(mask.detach() for mask in masks)
                    if bool(self.args.crs_detach_scale_evidence)
                    else masks
                ),
                pred,
                base_model.final,
            )
            if mode in (
                "crs_matched_random",
                "crs_same_pixel_random_scale",
                "crs_magnitude_nonpivotal",
            ):
                control_salt = (
                    int(getattr(self.args, "seed", 0)) * 1_000_003
                    + int(epoch) * 9_176
                    + int(getattr(self, "current_batch_index", 0))
                )
                control_function = (
                    matched_random_responsibility_suppression
                    if mode == "crs_matched_random"
                    else (
                        same_pixel_random_scale_suppression
                        if mode == "crs_same_pixel_random_scale"
                        else magnitude_matched_nonpivotal_suppression
                    )
                )
                responsibility_loss, responsibility_log = (
                    control_function(
                        pred,
                        coalition["contributions"],
                        labels,
                        safe_kernel=int(self.args.crs_safe_kernel),
                        normalization=getattr(
                            self.args, "sdrr_normalization", "event"
                        ),
                        **(
                            {}
                            if mode == "crs_magnitude_nonpivotal"
                            else {"salt": control_salt}
                        ),
                    )
                )
            else:
                responsibility_loss, responsibility_log = (
                    counterfactual_responsibility_suppression(
                        pred,
                        coalition["contributions"],
                        labels,
                        safe_kernel=int(self.args.crs_safe_kernel),
                        normalization=getattr(
                            self.args, "sdrr_normalization", "event"
                        ),
                    )
                )
            weighted = float(self.args.crs_lambda) * ramp * responsibility_loss
            self.last_deep_supervision_log.update({
                "canonical_loss": canonical_total.detach(),
                "crs_loss_raw": responsibility_loss.detach(),
                "crs_loss_weighted": weighted.detach(),
                "crs_ramp": ramp,
                "counterfactual_responsibility": float(
                    mode == "crs_flip_suppression"
                ),
                "scale_budget_random_control": float(
                    mode == "crs_matched_random"
                ),
                "same_pixel_random_scale_control": float(
                    mode == "crs_same_pixel_random_scale"
                ),
                "magnitude_matched_nonpivotal_control": float(
                    mode == "crs_magnitude_nonpivotal"
                ),
                "crs_detach_scale_evidence": float(
                    bool(self.args.crs_detach_scale_evidence)
                ),
                **responsibility_log,
            })
            return canonical_total + weighted

        if is_tfds_deep_supervision(mode):
            if instance_map is None:
                raise RuntimeError("%s requires return_instance_map batches" % mode)
            assignment = self.task_consistent_graph(instance_map)
            side_losses = []
            log_vars = {}
            for side_index, side_logit in enumerate(masks):
                target, valid, active = self.task_consistent_target_builder(
                    instance_map,
                    assignment,
                    side_index,
                )
                per_sample_loss = self.partial_sls_loss(
                    side_logit,
                    target,
                    valid,
                    self.warm_epoch,
                    epoch,
                    reduction="none",
                )
                all_valid_sample = (valid == 1).flatten(1).all(dim=1)
                gradient_active = active | all_valid_sample
                if mode == "tfds_projection_active_renorm":
                    # Diagnostic definition: renormalize only sample-head pairs
                    # that retain at least one known positive.  Canonical empty
                    # crops remain visible in gradient_active_ratio but are not
                    # counted in the positive-supervision budget.
                    active_float = active.to(per_sample_loss.dtype)
                    active_count = active_float.sum()
                    if bool((active_count > 0).detach().cpu()):
                        side_loss = (
                            per_sample_loss * active_float
                        ).sum() / active_count
                    else:
                        side_loss = per_sample_loss.sum() * 0.0
                else:
                    side_loss = per_sample_loss.mean()
                side_losses.append(side_loss)
                graph_values = [
                    graph[:, side_index].float()
                    for graph in assignment.feasible
                    if graph.numel() > 0
                ]
                feasible_ratio = (
                    torch.cat(graph_values).mean()
                    if graph_values
                    else side_logit.new_zeros(())
                )
                log_vars["side%d_loss" % side_index] = side_loss.detach()
                log_vars["side%d_positive_active_ratio" % side_index] = (
                    active.float().mean()
                )
                log_vars["side%d_gradient_active_ratio" % side_index] = (
                    gradient_active.float().mean()
                )
                log_vars["side%d_valid_ratio" % side_index] = valid.mean()
                log_vars["side%d_unknown_ratio" % side_index] = 1.0 - valid.mean()
                log_vars["side%d_positive_pixel_ratio" % side_index] = target.mean()
                log_vars["side%d_feasible_ratio" % side_index] = feasible_ratio.detach()

            # Preserve canonical MSHNet's exact five-objective aggregation.
            total = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            self.last_deep_supervision_log.update(log_vars)
            self.last_deep_supervision_log["canonical_aggregation"] = 1.0
            self.last_deep_supervision_log["active_renorm"] = float(
                mode == "tfds_projection_active_renorm"
            )
            return total

        if is_tgds_deep_supervision(mode):
            labels_for_scale = labels
            side_losses = []
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    labels_for_scale = self.down(labels_for_scale)
                side_loss = self.loss_fun(
                    side_logit,
                    labels_for_scale,
                    self.warm_epoch,
                    epoch,
                )
                side_losses.append(side_loss)
                self.last_deep_supervision_log[
                    "side%d_loss" % side_index
                ] = side_loss.detach()

            total = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            self.last_tgds_components = (final_loss, tuple(side_losses))
            self.last_deep_supervision_log["canonical_aggregation"] = 1.0
            self.last_deep_supervision_log["task_halfspace"] = 1.0
            return total

        if is_tsds_deep_supervision(mode):
            side_losses = []
            for side_index, side_logit in enumerate(masks):
                lifted_logit = (
                    side_logit
                    if side_logit.shape[-2:] == labels.shape[-2:]
                    else F.interpolate(
                        side_logit,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                side_loss = self.loss_fun(
                    lifted_logit,
                    labels,
                    self.warm_epoch,
                    epoch,
                )
                side_losses.append(side_loss)
                self.last_deep_supervision_log[
                    "side%d_task_loss" % side_index
                ] = side_loss.detach()
            total = (final_loss + torch.stack(side_losses).sum()) / (
                len(side_losses) + 1
            )
            self.last_deep_supervision_log["canonical_aggregation"] = 1.0
            self.last_deep_supervision_log["task_space_lift"] = 1.0
            return total

        if is_prds_deep_supervision(mode):
            projected_label = labels
            side_regrets = []
            full_valid = torch.ones_like(labels)
            for side_index, side_logit in enumerate(masks):
                if side_index > 0:
                    projected_label = self.down(projected_label)
                lifted_logit = (
                    side_logit
                    if side_logit.shape[-2:] == labels.shape[-2:]
                    else F.interpolate(
                        side_logit,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                oracle_lift = (
                    projected_label
                    if projected_label.shape[-2:] == labels.shape[-2:]
                    else F.interpolate(
                        projected_label,
                        size=labels.shape[-2:],
                        mode="nearest",
                    )
                )
                oracle_logit = torch.where(
                    oracle_lift > 0.5,
                    torch.full_like(oracle_lift, 12.0),
                    torch.full_like(oracle_lift, -12.0),
                )
                current_terms = self.partial_sls_loss(
                    lifted_logit,
                    labels,
                    full_valid,
                    self.warm_epoch,
                    epoch,
                    reduction="none",
                )
                oracle_terms = self.partial_sls_loss(
                    oracle_logit,
                    labels,
                    full_valid,
                    self.warm_epoch,
                    epoch,
                    reduction="none",
                ).detach()
                regret_terms = torch.relu(current_terms - oracle_terms)
                side_regret = regret_terms.mean()
                side_regrets.append(side_regret)
                self.last_deep_supervision_log.update({
                    "side%d_regret" % side_index: side_regret.detach(),
                    "side%d_regret_active_ratio" % side_index: (
                        regret_terms > 0
                    ).float().mean(),
                    "side%d_oracle_floor" % side_index: oracle_terms.mean(),
                    "side%d_task_loss" % side_index: current_terms.mean().detach(),
                })
            total = (final_loss + torch.stack(side_regrets).sum()) / (
                len(side_regrets) + 1
            )
            self.last_deep_supervision_log["canonical_aggregation"] = 1.0
            self.last_deep_supervision_log["projection_regret"] = 1.0
            return total

        if is_coalition_deep_supervision(mode):
            base_model = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            if mode == "sfds_filtration":
                filtration = nested_scale_filtration(
                    masks,
                    pred,
                    base_model.final,
                )
                filtration_losses = []
                for terminal_scale in range(4):
                    filtration_logit = filtration["filtration_logits"][
                        :, terminal_scale : terminal_scale + 1
                    ]
                    filtration_loss = self.loss_fun(
                        filtration_logit,
                        labels,
                        self.warm_epoch,
                        epoch,
                    )
                    filtration_losses.append(filtration_loss)
                    self.last_deep_supervision_log[
                        "prefix0to%d_loss" % terminal_scale
                    ] = filtration_loss.detach()
                total = torch.stack(filtration_losses).mean()
                reconstruction_error = (
                    filtration["reconstructed"] - pred
                ).abs().max()
                self.last_deep_supervision_log.update({
                    "filtration_reconstruction_error": (
                        reconstruction_error.detach()
                    ),
                    "nested_scale_filtration": 1.0,
                    "objective_count": 4.0,
                })
                return total

            if mode in ("asfs_anchor_filtration", "rdfs_continuation"):
                canonical_total = None
                rdfs_alpha = 1.0
                if mode == "rdfs_continuation":
                    rdfs_alpha = self.get_rdfs_alpha(epoch)
                    canonical_sum = final_loss
                    labels_for_scale = labels
                    for side_index, side_logit in enumerate(masks):
                        if side_index > 0:
                            labels_for_scale = self.down(labels_for_scale)
                        canonical_sum = canonical_sum + self.loss_fun(
                            side_logit,
                            labels_for_scale,
                            self.warm_epoch,
                            epoch,
                        )
                    canonical_total = canonical_sum / (len(masks) + 1)
                    if rdfs_alpha == 0.0:
                        self.last_deep_supervision_log.update({
                            "rdfs_alpha": 0.0,
                            "role_discovery": 1.0,
                            "canonical_loss": canonical_total.detach(),
                        })
                        return canonical_total

                filtration = nested_scale_filtration(
                    masks,
                    pred,
                    base_model.final,
                )
                side0_loss = self.loss_fun(
                    masks[0], labels, self.warm_epoch, epoch
                )
                side1_loss = self.loss_fun(
                    masks[1],
                    self.down(labels),
                    self.warm_epoch,
                    epoch,
                )
                prefix1_loss = self.loss_fun(
                    filtration["filtration_logits"][:, 1:2],
                    labels,
                    self.warm_epoch,
                    epoch,
                )
                prefix2_loss = self.loss_fun(
                    filtration["filtration_logits"][:, 2:3],
                    labels,
                    self.warm_epoch,
                    epoch,
                )
                terms = (
                    side0_loss,
                    side1_loss,
                    prefix1_loss,
                    prefix2_loss,
                    final_loss,
                )
                anchor_total = torch.stack(terms).mean()
                total = anchor_total
                if mode == "rdfs_continuation":
                    total = (
                        (1.0 - rdfs_alpha) * canonical_total
                        + rdfs_alpha * anchor_total
                    )
                reconstruction_error = (
                    filtration["reconstructed"] - pred
                ).abs().max()
                self.last_deep_supervision_log.update({
                    "anchor_side0_loss": side0_loss.detach(),
                    "anchor_side1_loss": side1_loss.detach(),
                    "prefix0to1_loss": prefix1_loss.detach(),
                    "prefix0to2_loss": prefix2_loss.detach(),
                    "anchor_filtration_reconstruction_error": (
                        reconstruction_error.detach()
                    ),
                    "anchor_filtration": 1.0,
                    "canonical_objective_count": 5.0,
                })
                if mode == "rdfs_continuation":
                    self.last_deep_supervision_log.update({
                        "rdfs_alpha": rdfs_alpha,
                        "role_discovery": float(rdfs_alpha < 1.0),
                        "canonical_loss": canonical_total.detach(),
                        "anchor_filtration_loss": anchor_total.detach(),
                    })
                return total

            coalition = leave_one_scale_out_coalitions(
                masks, pred, base_model.final
            )
            coalition_losses = []
            for deleted_scale in range(4):
                coalition_logit = coalition["coalition_logits"][
                    :, deleted_scale : deleted_scale + 1
                ]
                coalition_loss = self.loss_fun(
                    coalition_logit,
                    labels,
                    self.warm_epoch,
                    epoch,
                )
                coalition_losses.append(coalition_loss)
                self.last_deep_supervision_log[
                    "without_scale%d_loss" % deleted_scale
                ] = coalition_loss.detach()
            total = (
                final_loss + torch.stack(coalition_losses).sum()
            ) / 5.0
            reconstruction_error = (
                coalition["reconstructed"] - pred
            ).abs().max()
            self.last_deep_supervision_log.update({
                "coalition_reconstruction_error": reconstruction_error.detach(),
                "counterfactual_coalition": 1.0,
                "canonical_objective_count": 5.0,
            })
            return total

        if mode in ("legacy_rescaled", "side_no_location"):
            labels_for_scale = labels
            aux_losses = []
            for j, side_logit in enumerate(masks):
                if j > 0:
                    labels_for_scale = self.down(labels_for_scale)
                aux_losses.append(
                    self.loss_fun(
                        side_logit,
                        labels_for_scale,
                        self.warm_epoch,
                        epoch,
                        with_shape=(mode == "legacy_rescaled"),
                    )
                )
            aux_loss = torch.stack(aux_losses).mean()
            self.last_deep_supervision_log.update(
                {
                    "aux_loss_raw": aux_loss.detach(),
                    "aux_loss_weighted": (
                        self.args.aux_loss_weight * aux_loss
                    ).detach(),
                }
            )
            return final_loss + self.args.aux_loss_weight * aux_loss

        if not is_rods_deep_supervision(mode):
            raise RuntimeError("unsupported deep supervision mode: %s" % mode)
        if instance_map is None:
            raise RuntimeError("%s requires return_instance_map batches" % mode)

        assignment = self.resolution_graph(instance_map)
        aux_terms = []
        log_vars = {}
        for side_index, side_logit in enumerate(masks):
            target, valid, weight, active = self.owned_supervision_builder(
                instance_map,
                assignment,
                side_index,
            )
            per_sample = self.masked_owned_loss(side_logit, target, valid, weight)
            if self.args.empty_side_policy == "skip":
                active = active.to(dtype=per_sample.dtype)
                active_sum = active.sum()
                if bool((active_sum > 0).detach().cpu()):
                    side_loss = (per_sample * active).sum() / active_sum.clamp_min(1.0)
                else:
                    side_loss = per_sample.sum() * 0.0
            else:
                side_loss = per_sample.mean()
            aux_terms.append(side_loss)
            log_vars["side%d_loss" % side_index] = side_loss.detach()
            log_vars["side%d_active_ratio" % side_index] = active.float().mean().detach()
            log_vars["side%d_valid_ratio" % side_index] = valid.detach().mean()
            log_vars["side%d_pos_ratio" % side_index] = target.detach().mean()

        aux_loss = torch.stack(aux_terms).mean()
        self.last_deep_supervision_log.update(
            {
                "aux_loss_raw": aux_loss.detach(),
                "aux_loss_weighted": (
                    self.args.aux_loss_weight * aux_loss
                ).detach(),
            }
        )
        self.last_deep_supervision_log.update(log_vars)
        return final_loss + self.args.aux_loss_weight * aux_loss

    def backward_task_gradient_supervision(self):
        """Apply the asymmetric TGDS constraint in parameter space."""

        if self.last_tgds_components is None:
            raise RuntimeError("TGDS backward requires stored loss components")
        final_loss, side_losses = self.last_tgds_components
        parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        task_gradient = torch.autograd.grad(
            final_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        projected_auxiliary = []
        for side_index, side_loss in enumerate(side_losses):
            auxiliary_gradient = torch.autograd.grad(
                side_loss,
                parameters,
                retain_graph=side_index + 1 < len(side_losses),
                create_graph=False,
                allow_unused=True,
            )
            projected, statistics = project_auxiliary_gradient(
                task_gradient,
                auxiliary_gradient,
            )
            projected_auxiliary.append(projected)
            post_inner = gradient_inner_product(task_gradient, projected)
            self.last_deep_supervision_log.update({
                "side%d_gradient_cosine" % side_index: statistics["cosine"],
                "side%d_conflict" % side_index: statistics["conflict"],
                "side%d_task_grad_norm" % side_index: statistics["task_norm"],
                "side%d_aux_grad_norm" % side_index: statistics["auxiliary_norm"],
                "side%d_removed_grad_norm" % side_index: statistics["removed_norm"],
                "side%d_post_inner" % side_index: post_inner.detach(),
            })

        combined = combine_task_and_auxiliary_gradients(
            task_gradient,
            projected_auxiliary,
            denominator=float(len(side_losses) + 1),
        )
        for parameter, gradient in zip(parameters, combined):
            parameter.grad = None if gradient is None else gradient.detach()

    def train(self, epoch):
        self.model.train()
        self.configure_full_dea_trainable(epoch)
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        for i, batch in enumerate(tbar):
            self.current_batch_index = i
            data, mask, instance_map = self.unpack_batch(batch)
  
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)
            if instance_map is not None:
                instance_map = instance_map.to(self.device, non_blocking=True)

            tag = self.get_forward_tag(epoch)
            use_dea = self.use_dea(epoch)

            full_dea_out = None
            dea_main_out = None
            cev_out = None
            if is_cev_control(self.args.model_type):
                out = self.model(data, True, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                cev_out = out["cev"]
                dea_out = None
            elif self.args.model_type == "full_dea":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                full_dea_out = out["full_dea"]
                dea_out = None
            elif self.args.model_type == "dea_integrated":
                out = self.model(data, tag, return_dict=True)
                masks = out["masks"]
                pred = out["pred"]
                dea_out = None
            elif is_dea_main_model(self.args.model_type):
                return_details = (
                    self.args.predictive_log_interval > 0
                    and i % self.args.predictive_log_interval == 0
                )
                dea_main_out = self.model(
                    data,
                    tag,
                    return_dict=True,
                    return_details=return_details,
                )
                masks = []
                pred = dea_main_out["pred"]
                dea_out = None
            elif use_dea:
                masks, pred, dea_out = self.model(
                    data,
                    tag,
                    return_dea=True,
                    dea_detach_evidence=self.args.dea_detach_evidence,
                )
            else:
                fusion_alpha = (
                    self.get_hms_alpha(epoch)
                    if is_homotopy_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    else None
                )
                if fusion_alpha is None:
                    # Preserve the physically isolated canonical MSHNet
                    # two-argument interface.  Experimental workbench-only
                    # keywords must not leak into baseline/SDRR forwards.
                    masks, pred = self.model(data, tag)
                else:
                    masks, pred = self.model(
                        data,
                        tag,
                        fusion_alpha=fusion_alpha,
                    )
                dea_out = None

            if cev_out is not None:
                # Side heads and the whole MSHNet path are frozen.  Averaging
                # their constant auxiliary losses with the only trainable final
                # CEV loss would silently divide the veto gradient by five.
                loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)
            elif dea_main_out is not None:
                loss = self.dea_main_loss(
                    dea_main_out["state_logits"], labels, epoch
                )
            else:
                loss = self.compute_deep_supervision_loss(
                    pred,
                    masks,
                    labels,
                    instance_map,
                    epoch,
                )
            loss_seg_for_debug = loss.detach()
            tgds_backward_done = False
            if is_tgds_deep_supervision(
                getattr(self.args, "deep_supervision", "")
            ):
                self.optimizer.zero_grad(set_to_none=True)
                self.backward_task_gradient_supervision()
                tgds_backward_done = True
            if (
                (
                    is_rods_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_tfds_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_tgds_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_tsds_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_prds_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_coalition_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_responsibility_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_scale_subset_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_scheduled_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_homotopy_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    or is_measure_conditioned_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                )
                and self.args.rods_log_interval > 0
                and i % self.args.rods_log_interval == 0
            ):
                msg = self.format_log_dict(self.last_deep_supervision_log)
                prefix = (
                    '[TCDS] '
                    if is_tfds_deep_supervision(
                        getattr(self.args, "deep_supervision", "")
                    )
                    else (
                        '[TGDS] '
                        if is_tgds_deep_supervision(
                            getattr(self.args, "deep_supervision", "")
                        )
                        else (
                            '[TSDS] '
                            if is_tsds_deep_supervision(
                                getattr(self.args, "deep_supervision", "")
                            )
                            else (
                                '[PRDS] '
                                if is_prds_deep_supervision(
                                    getattr(self.args, "deep_supervision", "")
                                )
                                else (
                                    '[CSCS] '
                                    if is_coalition_deep_supervision(
                                        getattr(self.args, "deep_supervision", "")
                                    )
                                    else (
                                        '[CRS] '
                                        if is_responsibility_deep_supervision(
                                            getattr(self.args, "deep_supervision", "")
                                        )
                                        else (
                                            '[SCALE CONTROL] '
                                            if (
                                                is_scale_subset_deep_supervision(
                                                    getattr(self.args, "deep_supervision", "")
                                                )
                                                or is_scheduled_deep_supervision(
                                                    getattr(self.args, "deep_supervision", "")
                                                )
                                            )
                                            else (
                                                '[HMS] '
                                                if is_homotopy_deep_supervision(
                                                    getattr(self.args, "deep_supervision", "")
                                                )
                                                else (
                                                    '[MC-SLS] '
                                                    if is_measure_conditioned_deep_supervision(
                                                        getattr(self.args, "deep_supervision", "")
                                                    )
                                                    else '[RODS] '
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                print(prefix + ' | '.join(msg))
            integrated_route_loss = None
            integrated_route_log = {}
            integrated_route_ramp = 0.0
            if (
                self.args.model_type == "dea_integrated"
                and self.args.integrated_route_loss_weight > 0.0
            ):
                integrated_route_loss, integrated_route_log = (
                    residual_aligned_route_loss(out, labels)
                )
                if self.args.integrated_route_ramp_epochs > 0:
                    integrated_route_ramp = min(
                        1.0,
                        float(epoch + 1)
                        / float(self.args.integrated_route_ramp_epochs),
                    )
                else:
                    integrated_route_ramp = 1.0
                loss = loss + (
                    self.args.integrated_route_loss_weight
                    * integrated_route_ramp
                    * integrated_route_loss
                )

            if self.args.model_type == "full_dea" and full_dea_out is not None:
                ramp = self.get_full_dea_ramp(epoch)
                if self.args.full_dea_version == "v2":
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v2(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        topk_ratio=self.args.full_dea_topk_ratio,
                        topk_min_score=self.args.full_dea_topk_min_score,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                elif self.args.full_dea_version == "v3":
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v3(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        protect_kernel=self.args.full_dea_protect_kernel,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        min_component_area=self.args.full_dea_hard_min_area,
                        max_component_area=self.args.full_dea_hard_max_area,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                else:
                    loss_full_dea, full_dea_log = full_dea_aux_loss_v4(
                        full_dea_out=full_dea_out,
                        target=labels,
                        epoch=epoch,
                        warm_epoch=self.warm_epoch,
                        seg_criterion=self.loss_fun,
                        tau_base=self.args.full_dea_tau_base,
                        tau_target=self.args.full_dea_tau_target,
                        tau_scale=self.args.full_dea_tau_scale,
                        protect_kernel=self.args.full_dea_protect_kernel,
                        safe_kernel=self.args.full_dea_safe_kernel,
                        min_component_area=self.args.full_dea_hard_min_area,
                        max_component_area=self.args.full_dea_hard_max_area,
                        max_hard_bg_ratio=self.args.full_dea_max_hard_bg_ratio,
                    )
                loss = loss + self.args.full_dea_lambda * ramp * loss_full_dea

                if self.args.full_dea_debug and i % max(1, self.args.dea_debug_interval) == 0:
                    msg = [
                        'full_dea_ramp=%.6f' % ramp,
                        'full_dea_loss=%.6f' % float(loss_full_dea.detach()),
                        'full_dea_weighted=%.6f'
                        % float((self.args.full_dea_lambda * ramp * loss_full_dea).detach()),
                    ]
                    msg.extend(self.format_log_dict(full_dea_log))
                    print('[FULL DEA DEBUG] ' + ' | '.join(msg))
            elif use_dea:
                ramp = get_dea_ramp(epoch, self.warm_epoch, self.args.dea_ramp_epochs)
                cur_lambda_single = self.args.dea_lambda_single * ramp
                cur_lambda_dec = self.args.dea_lambda_dec * ramp
                cur_lambda_empty = self.args.dea_lambda_empty * ramp

                loss_dea, dea_log = dea_lite_loss(
                    dea_out=dea_out,
                    z_full=pred,
                    gt=labels,
                    lambda_single=cur_lambda_single,
                    lambda_dec=cur_lambda_dec,
                    lambda_empty=cur_lambda_empty,
                    tau=self.args.dea_tau,
                )
                loss = loss + loss_dea
                self.save_dea_debug(epoch, i, data, labels, pred, dea_out)

                if self.args.save_dea_debug and self.args.dea_debug_interval > 0 and i % self.args.dea_debug_interval == 0:
                    dea_ratio = (loss_dea.detach() / (loss_seg_for_debug + 1e-6)).item()
                    msg = [
                        'dea_ratio=%.4f' % dea_ratio,
                        'lambda_single=%.6f' % cur_lambda_single,
                        'lambda_empty=%.6f' % cur_lambda_empty,
                        'lambda_dec=%.6f' % cur_lambda_dec,
                    ]
                    msg.extend(self.format_log_dict(dea_log))
                    print('[DEA DEBUG] ' + ' | '.join(msg))

            if (
                self.args.model_type == "dea_integrated"
                and self.args.integrated_log_interval > 0
                and i % self.args.integrated_log_interval == 0
            ):
                core_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                route_stats = core_model.route_statistics(out["routes"])
                msg = self.format_log_dict(route_stats)
                if "scale_fusion" in out:
                    deltas = out["scale_fusion"]["deltas"]
                    msg.append('scale_delta_abs=%.6f' % float(deltas.detach().abs().mean()))
                if integrated_route_loss is not None:
                    weighted_route_loss = (
                        self.args.integrated_route_loss_weight
                        * integrated_route_ramp
                        * integrated_route_loss.detach()
                    )
                    msg.extend([
                        'route_ramp=%.4f' % integrated_route_ramp,
                        'route_loss_raw=%.6f' % float(integrated_route_loss.detach()),
                        'route_loss_weighted=%.6f' % float(weighted_route_loss),
                        'route_to_seg=%.6f'
                        % float(weighted_route_loss / (loss_seg_for_debug + 1e-6)),
                    ])
                    msg.extend(self.format_log_dict(integrated_route_log))
                print('[INTEGRATED DEA] ' + ' | '.join(msg))

            if (
                dea_main_out is not None
                and "corrections" in dea_main_out
            ):
                core_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                stats = core_model.state_statistics(dea_main_out)
                print('[DEA MAIN] ' + ' | '.join(
                    self.format_log_dict(stats)
                ))
        
            if not tgds_backward_done:
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        # A saved model must always be evaluated through its complete
        # multi-scale inference graph.  Tying this to a training epoch silently
        # bypassed final fusion in the old test entry point.
        tag = True
        route_audit = (
            self.new_integrated_route_audit()
            if self.args.model_type == "dea_integrated"
            else None
        )
        with torch.no_grad():
            for i, batch in enumerate(tbar):
                data, mask, _instance_map = self.unpack_batch(batch)
    
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                loss = 0
                if self.args.model_type in (
                    "full_dea",
                    "dea_integrated",
                ) or is_dea_main_model(self.args.model_type) or is_cev_control(
                    self.args.model_type
                ):
                    out = self.model(data, tag, return_dict=True)
                    pred = out["pred"]
                    if route_audit is not None:
                        self.update_integrated_route_audit(route_audit, out, mask)
                else:
                    _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()
            if route_audit is not None:
                self.finalize_integrated_route_audit(route_audit, epoch)

            
            if self.mode == 'train':
                current_pd = PD[0]
                current_fa = FA[0] * 1000000
                if self.args.paired_baseline_iou > 0:
                    pd_fa_iou_threshold = max(
                        self.args.pd_fa_min_iou,
                        self.args.paired_baseline_iou - self.args.pd_fa_iou_margin,
                    )
                else:
                    pd_fa_iou_threshold = self.args.pd_fa_min_iou

                is_pd_fa_candidate = (
                    current_pd >= self.args.pd_fa_min_pd
                    and mean_IoU >= pd_fa_iou_threshold
                    and current_fa < self.best_pd_fa
                )
                metric_line = '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                    epoch,
                    mean_IoU,
                    current_pd,
                    current_fa,
                )
                print(metric_line.strip())
                with open(osp.join(self.save_folder, 'epoch_metric.log'), 'a') as f:
                    f.write(metric_line)

                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight.pkl'),
                    )

                    best_iou_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                        "method_meta": get_method_metadata(self.args),
                    }
                    torch.save(
                        best_iou_states,
                        osp.join(self.save_folder, 'checkpoint_best_iou.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, current_pd, current_fa))

                if is_pd_fa_candidate:
                    self.best_pd_fa = current_fa
                    self.best_pd_fa_iou = mean_IoU
                    self.best_pd_fa_pd = current_pd
                    self.best_pd_fa_epoch = epoch

                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight_pd_fa_best.pkl'),
                    )

                    pd_fa_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                        "method_meta": get_method_metadata(self.args),
                    }
                    torch.save(
                        pd_fa_states,
                        osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, mean_IoU, current_pd, current_fa))
                        
                latest_states = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "iou": mean_IoU,
                    "pd": current_pd,
                    "fa": current_fa,
                    "best_iou": self.best_iou,
                    "best_pd_fa": self.best_pd_fa,
                    "best_pd_fa_iou": self.best_pd_fa_iou,
                    "best_pd_fa_pd": self.best_pd_fa_pd,
                    "best_pd_fa_epoch": self.best_pd_fa_epoch,
                    "method_meta": get_method_metadata(self.args),
                }
                torch.save(latest_states, osp.join(self.save_folder, 'checkpoint.pkl'))
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed, args.deterministic)

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 
