from pathlib import Path

from PIL import Image

from tools.audit_dataset_pair_integrity import audit_dataset


def _save(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size).save(path)


def test_integrity_audit_distinguishes_rescalable_and_invalid_pairs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "SIRST-v1"
    manifest = dataset / "img_idx" / "test_SIRST-v1.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("exact\nrescale\nbad\n")
    _save(dataset / "images" / "exact.png", (20, 10))
    _save(dataset / "masks" / "exact.png", (20, 10))
    _save(dataset / "images" / "rescale.png", (40, 20))
    _save(dataset / "masks" / "rescale.png", (80, 40))
    _save(dataset / "images" / "bad.png", (20, 10))
    _save(dataset / "masks" / "bad.png", (20, 20))

    result = audit_dataset(dataset)

    assert result["exact_geometry_pairs"] == 1
    assert [item["id"] for item in result["same_aspect_rescalable_pairs"]] == [
        "rescale"
    ]
    assert [item["id"] for item in result["invalid_geometry_pairs"]] == ["bad"]
    assert not result["pass"]
