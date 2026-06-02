from pathlib import Path

import cv2
import numpy as np

from prepare_visdrone import convert_visdrone_split, materialize_image, resolve_split_dirs, write_dataset_yaml


def test_convert_visdrone_split_writes_yolo_labels_and_skips_ignored(tmp_path: Path):
    source = tmp_path / "raw" / "VisDrone2019-DET-train"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(source / "images" / "000001.jpg"), image)
    (source / "annotations" / "000001.txt").write_text(
        "\n".join(
            [
                "10,20,30,40,1,1,0,0",
                "50,60,70,80,0,3,0,0",
            ]
        ),
        encoding="utf-8",
    )

    summary = convert_visdrone_split(source, tmp_path / "prepared", split_name="train", copy_images=True)

    label = (tmp_path / "prepared" / "labels" / "train" / "000001.txt").read_text(encoding="utf-8").strip()
    assert label == "0 0.250000 0.400000 0.300000 0.400000"
    assert (tmp_path / "prepared" / "images" / "train" / "000001.jpg").exists()
    assert summary["images"] == 1
    assert summary["boxes"] == 1
    assert summary["ignored"] == 1


def test_write_dataset_yaml_points_to_prepared_output(tmp_path: Path):
    yaml_path = write_dataset_yaml(tmp_path / "prepared", tmp_path / "prepared" / "VisDrone-prepared.yaml")

    text = yaml_path.read_text(encoding="utf-8")

    assert f"path: {tmp_path / 'prepared'}" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "0: pedestrian" in text


def test_resolve_split_dirs_handles_nested_kaggle_layout(tmp_path: Path):
    split = tmp_path / "VisDrone2019-DET-train"
    nested = split / "VisDrone2019-DET-train"
    (nested / "images").mkdir(parents=True)
    (nested / "annotations").mkdir()

    image_dir, annotation_dir = resolve_split_dirs(split)

    assert image_dir == nested / "images"
    assert annotation_dir == nested / "annotations"


def test_resolve_split_dirs_handles_images_directly_in_split(tmp_path: Path):
    split = tmp_path / "VisDrone2019-DET-val"
    split.mkdir()
    cv2.imwrite(str(split / "000001.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    (split / "annotations").mkdir()

    image_dir, annotation_dir = resolve_split_dirs(split)

    assert image_dir == split
    assert annotation_dir == split / "annotations"


def test_materialize_image_symlink_falls_back_to_copy(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "nested" / "dst.jpg"
    src.write_bytes(b"image-bytes")

    def fail_symlink(_src, _dst):
        raise OSError("symlink unavailable")

    monkeypatch.setattr("os.symlink", fail_symlink)

    mode_used = materialize_image(src, dst, image_mode="symlink")

    assert mode_used == "copy"
    assert dst.read_bytes() == b"image-bytes"
