from pathlib import Path

import cv2
import numpy as np

from prepare_visdrone import convert_visdrone_split, write_dataset_yaml


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
