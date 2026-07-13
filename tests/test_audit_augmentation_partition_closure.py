from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.audit_augmentation_partition_closure import (
    PROJECT_ROOT,
    WITNESSES,
    _edge_rectangles,
    audit_dataset,
    coverage_from_rectangles,
    greedy_rectangle_certificate,
    pil_mask_to_binary,
    resize_and_pad_mask,
    transform_mask,
)
from utils.data import IRSTD_Dataset


def _toy_mask() -> Image.Image:
    array = np.zeros((256, 256), dtype=np.uint8)
    array[127:130, 126:131] = 255
    return Image.fromarray(array, mode="L")


def test_transform_matches_repository_for_explicit_random_draws() -> None:
    mask = _toy_mask()
    image = Image.merge("RGB", (mask, mask, mask))
    dataset = object.__new__(IRSTD_Dataset)
    dataset.base_size = 256
    dataset.crop_size = 256

    # random.random: flip, then blur. random.randint: long size, crop x, crop y.
    with patch("utils.data.random.random", side_effect=[0.1, 0.9]), patch(
        "utils.data.random.randint", side_effect=[512, 37, 149]
    ):
        _, repository_mask = dataset._sync_transform(image, mask)

    audited = transform_mask(
        mask,
        long_size=512,
        flip=True,
        crop_x=37,
        crop_y=149,
    )
    assert np.array_equal(audited, pil_mask_to_binary(repository_mask))


def test_symbolic_pair_rectangles_cover_the_complete_grid() -> None:
    array = np.zeros((512, 512), dtype=np.uint8)
    # These two pairs independently translate to every horizontal/vertical
    # output edge under crop offsets 0..256.
    array[255, 254:256] = 255
    array[254:256, 255] = 255
    binary = pil_mask_to_binary(Image.fromarray(array, mode="L"))
    rectangles = _edge_rectangles(binary, flip=False)

    horizontal = coverage_from_rectangles(
        rectangles["horizontal"], orientation="horizontal"
    )
    vertical = coverage_from_rectangles(
        rectangles["vertical"], orientation="vertical"
    )
    assert horizontal.shape == (256, 255)
    assert vertical.shape == (255, 256)
    assert horizontal.all()
    assert vertical.all()
    assert len(
        greedy_rectangle_certificate(
            rectangles["horizontal"], orientation="horizontal"
        )
    ) == 1
    assert len(
        greedy_rectangle_certificate(
            rectangles["vertical"], orientation="vertical"
        )
    ) == 1


def test_padding_matches_repository_bottom_right_convention() -> None:
    mask = Image.fromarray(np.full((100, 200), 255, dtype=np.uint8), mode="L")
    transformed = resize_and_pad_mask(mask, long_size=128, flip=False)
    assert transformed.size == (256, 256)
    binary = pil_mask_to_binary(transformed)
    assert binary[:64, :128].all()
    assert not binary[64:, :].any()
    assert not binary[:, 128:].any()


def test_frozen_witnesses_are_train_only_and_certify_each_dataset() -> None:
    datasets_root = PROJECT_ROOT / "datasets"
    expected = {
        "NUAA-SIRST": (("Misc_11", "Misc_119"), 2),
        "NUDT-SIRST": (("000848",), 2),
        "IRSTD-1K": (("XDU730",), 2),
    }
    for dataset_name, (equality_names, components) in expected.items():
        result = audit_dataset(datasets_root / dataset_name)
        assert tuple(result["equality_witness"]["names"]) == equality_names
        assert result["inequality_witness"]["component_count"] == components
        assert not result["fixed_disjoint_coordinate_partition_exists"]
        assert result["internal_split"]["all_witnesses_are_fit_members"]
        for orientation in ("horizontal", "vertical"):
            edge_result = result["equality_witness"]["edge_coverage"][orientation]
            assert edge_result["reachable_output_edges"] == 65_280
            assert edge_result["all_output_edges"] == 65_280
            assert edge_result["complete"]
            assert edge_result["greedy_rectangle_certificate"]
        assert "/test" not in result["official_train_manifest"]
        for source in result["equality_witness"]["sources"]:
            assert "/test" not in source["mask_path"]
            assert source["image_mask_size_equal"]
            assert source["original_image_size_wh"] == source["original_size_wh"]
        assert "/test" not in result["inequality_witness"]["mask_path"]
        assert result["inequality_witness"]["image_mask_size_equal"]


def test_all_frozen_witness_names_are_explicit() -> None:
    assert WITNESSES == {
        "NUAA-SIRST": {
            "equality_names": ("Misc_11", "Misc_119"),
            "equality_flips": (False, True),
            "inequality_name": "Misc_119",
        },
        "NUDT-SIRST": {
            "equality_names": ("000848",),
            "equality_flips": (False,),
            "inequality_name": "000848",
        },
        "IRSTD-1K": {
            "equality_names": ("XDU730",),
            "equality_flips": (False,),
            "inequality_name": "XDU907",
        },
    }
