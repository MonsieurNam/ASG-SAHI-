import json
from pathlib import Path

import cv2
import numpy as np

from adaptive_sahi.pseudolabels import (
    PseudoLabelConfig,
    filter_pseudo_labels,
    xyxy_to_yolo_row,
)
from build_distilled_dataset import build_distilled_dataset


def test_filter_keeps_only_whitelisted_confident_small_boxes():
    teacher = np.array(
        [
            [10, 10, 30, 30, 0.50, 0],
            [40, 40, 60, 60, 0.20, 0],
            [70, 70, 90, 90, 0.90, 3],
            [0, 0, 80, 80, 0.90, 1],
        ],
        dtype=float,
    )
    kept, manifest, summary = filter_pseudo_labels(
        teacher,
        np.zeros((0, 6), dtype=float),
        image_width=100,
        image_height=100,
        image_id="a.jpg",
        config=PseudoLabelConfig(class_ids={0, 1}, min_conf=0.30, max_area_ratio=0.10),
    )

    assert kept.shape == (1, 6)
    assert int(kept[0, 5]) == 0
    assert manifest[0]["score"] == 0.5
    assert summary["dropped_low_conf"] == 1
    assert summary["dropped_class"] == 1
    assert summary["dropped_large_area"] == 1


def test_filter_drops_same_class_duplicates_and_cross_class_ambiguity():
    teacher = np.array(
        [
            [11, 11, 31, 31, 0.90, 0],
            [61, 61, 81, 81, 0.80, 0],
            [5, 60, 25, 80, 0.70, 0],
        ],
        dtype=float,
    )
    gt = np.array(
        [
            [10, 10, 30, 30, 1.0, 0],
            [60, 60, 80, 80, 1.0, 1],
        ],
        dtype=float,
    )

    kept, _, summary = filter_pseudo_labels(
        teacher,
        gt,
        image_width=100,
        image_height=100,
        image_id="a.jpg",
        config=PseudoLabelConfig(class_ids={0}, same_class_iou_drop=0.30, cross_class_iou_drop=0.40),
    )

    assert kept.tolist() == [[5.0, 60.0, 25.0, 80.0, 0.7, 0.0]]
    assert summary["dropped_same_class_gt"] == 1
    assert summary["dropped_cross_class_gt"] == 1


def test_filter_limits_kept_boxes_after_sorting_by_confidence():
    teacher = np.array(
        [
            [0, 0, 10, 10, 0.20, 0],
            [20, 0, 30, 10, 0.90, 0],
            [40, 0, 50, 10, 0.70, 0],
        ],
        dtype=float,
    )

    kept, _, summary = filter_pseudo_labels(
        teacher,
        np.zeros((0, 6), dtype=float),
        image_width=100,
        image_height=100,
        image_id="a.jpg",
        config=PseudoLabelConfig(class_ids={0}, min_conf=0.0, max_pseudo_per_image=2),
    )

    assert kept[:, 4].tolist() == [0.9, 0.7]
    assert summary["dropped_max_pseudo"] == 1


def test_xyxy_to_yolo_row_clamps_to_image_bounds():
    row = xyxy_to_yolo_row(np.array([-10, 10, 120, 40, 0.9, 2], dtype=float), 100, 50)

    assert row == "2 0.500000 0.500000 1.000000 0.600000"


def test_build_distilled_dataset_keeps_val_labels_unchanged(tmp_path):
    prepared = tmp_path / "prepared"
    for split in ["train", "val"]:
        (prepared / "images" / split).mkdir(parents=True)
        (prepared / "labels" / split).mkdir(parents=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert cv2.imwrite(str(prepared / "images" / split / "sample.jpg"), image)
        (prepared / "labels" / split / "sample.txt").write_text(
            "0 0.500000 0.500000 0.100000 0.100000",
            encoding="utf-8",
        )
    (prepared / "VisDrone-prepared.yaml").write_text("path: prepared\n", encoding="utf-8")
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        json.dumps(
            {
                "image_id": "sample.jpg",
                "detections": [[20, 20, 30, 30, 0.9, 0]],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "std"
    summary = build_distilled_dataset(
        prepared_root=prepared,
        teacher_preds=teacher,
        output_root=output,
        config=PseudoLabelConfig(class_ids={0}, min_conf=0.3),
        image_mode="none",
        limit=1,
    )

    assert summary["train_images"] == 1
    assert summary["kept_pseudo"] == 1
    assert (output / "labels" / "val" / "sample.txt").read_text(encoding="utf-8") == (
        prepared / "labels" / "val" / "sample.txt"
    ).read_text(encoding="utf-8")
    train_lines = (output / "labels" / "train" / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 2
    assert (output / "pseudo_manifest.jsonl").exists()
