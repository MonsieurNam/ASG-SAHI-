import json
from pathlib import Path

import numpy as np

from adaptive_sahi.evaluation import evaluate_detection_map
from adaptive_sahi.io import load_prediction_jsonl, load_yolo_labels_xyxy, save_prediction_jsonl


def test_load_yolo_labels_xyxy_converts_normalized_boxes(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame001.txt").write_text("2 0.500000 0.250000 0.200000 0.100000\n", encoding="utf-8")

    boxes = load_yolo_labels_xyxy(labels_dir / "frame001.txt", image_width=1000, image_height=500)

    expected = np.array([[400.0, 100.0, 600.0, 150.0, 1.0, 2]], dtype=float)
    np.testing.assert_allclose(boxes, expected)


def test_prediction_jsonl_roundtrip_preserves_schema(tmp_path: Path):
    prediction_path = tmp_path / "predictions.jsonl"
    predictions = {
        "image_a.jpg": np.array([[1.0, 2.0, 3.0, 4.0, 0.9, 1]], dtype=float),
        "image_b.jpg": np.zeros((0, 6), dtype=float),
    }
    metadata = {"mode": "fixed_sahi", "latency_ms": 12.5, "slices": 4}

    save_prediction_jsonl(prediction_path, predictions, per_image_metadata={"image_a.jpg": metadata})
    loaded, loaded_metadata = load_prediction_jsonl(prediction_path)

    assert set(loaded) == {"image_a.jpg", "image_b.jpg"}
    np.testing.assert_allclose(loaded["image_a.jpg"], predictions["image_a.jpg"])
    assert loaded["image_b.jpg"].shape == (0, 6)
    assert loaded_metadata["image_a.jpg"]["mode"] == "fixed_sahi"
    assert loaded_metadata["image_b.jpg"] == {}

    raw = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    assert set(raw[0]) == {"image_id", "detections", "metadata"}


def test_evaluate_detection_map_perfect_prediction_scores_one():
    ground_truth = {
        "image_a.jpg": np.array([[10.0, 10.0, 50.0, 50.0, 1.0, 0]], dtype=float),
    }
    predictions = {
        "image_a.jpg": np.array([[10.0, 10.0, 50.0, 50.0, 0.9, 0]], dtype=float),
    }

    result = evaluate_detection_map(predictions, ground_truth, class_ids=[0], iou_thresholds=[0.5])

    assert result["mAP50"] == 1.0
    assert result["mAP50_95"] == 1.0
    assert result["per_class"][0]["AP50"] == 1.0


def test_evaluate_detection_map_wrong_class_scores_zero():
    ground_truth = {
        "image_a.jpg": np.array([[10.0, 10.0, 50.0, 50.0, 1.0, 0]], dtype=float),
    }
    predictions = {
        "image_a.jpg": np.array([[10.0, 10.0, 50.0, 50.0, 0.9, 1]], dtype=float),
    }

    result = evaluate_detection_map(predictions, ground_truth, class_ids=[0, 1], iou_thresholds=[0.5])

    assert result["mAP50"] == 0.0
    assert result["per_class"][0]["AP50"] == 0.0
