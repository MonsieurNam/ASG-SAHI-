"""Evaluate saved ASG-SAHI prediction JSONL files against YOLO labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_sahi.evaluation import evaluate_detection_map, summarize_metadata
from adaptive_sahi.io import image_size, list_images, load_prediction_jsonl, load_yolo_labels_xyxy


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASG-SAHI prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--method", default=None, help="Method name for tables; defaults to prediction filename stem.")
    parser.add_argument("--class-names", default=",".join(VISDRONE_NAMES))
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, metadata = load_prediction_jsonl(args.predictions)
    images_root = Path(args.images_dir)
    image_paths = list_images(images_root)
    if not image_paths:
        raise SystemExit(f"No images found in {images_root}")
    ground_truth = {}
    eval_predictions = {}

    for image_path in image_paths:
        image_id = image_path.relative_to(images_root).as_posix()
        prediction = predictions.get(image_id)
        if prediction is None:
            prediction = predictions.get(image_path.name)
        if prediction is None:
            prediction = np.zeros((0, 6), dtype=float)
        eval_predictions[image_id] = prediction

        label_path = Path(args.labels_dir) / Path(image_id).with_suffix(".txt")
        if not label_path.exists():
            label_path = Path(args.labels_dir) / f"{Path(image_id).stem}.txt"
        width, height = image_size(image_path)
        ground_truth[image_id] = load_yolo_labels_xyxy(label_path, width, height)

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    class_ids = list(range(len(class_names)))
    result = evaluate_detection_map(eval_predictions, ground_truth, class_ids=class_ids)
    runtime = summarize_metadata(metadata)
    method = args.method or Path(args.predictions).stem
    rows = [
        {
            "method": method,
            "mAP50": result["mAP50"],
            "mAP50_95": result["mAP50_95"],
            **runtime,
        }
    ]
    per_class_rows = []
    for class_id, values in result["per_class"].items():
        per_class_rows.append(
            {
                "method": method,
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                **values,
            }
        )

    prefix = Path(args.output_prefix) if args.output_prefix else Path(args.predictions).with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = prefix.with_suffix(".summary.csv")
    class_csv = prefix.with_suffix(".per_class.csv")
    summary_json = prefix.with_suffix(".metrics.json")
    summary_md = prefix.with_suffix(".summary.md")

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    pd.DataFrame(per_class_rows).to_csv(class_csv, index=False)
    summary_json.write_text(
        json.dumps({"summary": rows[0], "per_class": per_class_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_table = _markdown_table(rows)
    summary_md.write_text(summary_table, encoding="utf-8")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-class CSV: {class_csv}")
    print(f"Metrics JSON: {summary_json}")
    print(summary_table)


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
