"""Run ASG-SAHI inference modes and save reproducible prediction JSONL files.

Example:
    python run_experiment.py --mode fixed_sahi --weights runs/train/exp/weights/best.pt \
        --source datasets/VisDrone/images/val --output runs/preds/fixed_sahi.jsonl --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_sahi.detectors import UltralyticsDetector
from adaptive_sahi.engine import InferenceConfig, VALID_MODES, run_image_inference
from adaptive_sahi.io import list_images, read_image, save_prediction_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full, fixed SAHI, and ASG-SAHI inference modes.")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), required=True)
    parser.add_argument("--weights", required=True, help="Path to YOLO best.pt.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--output", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="Detector-internal NMS IoU.")
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--preview-imgsz", type=int, default=640)
    parser.add_argument("--fixed-slice-size", type=int, default=640)
    parser.add_argument("--fixed-overlap", type=float, default=0.25)
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--postprocess", choices=["auto", "none", "nms", "wbf"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test image limit.")
    return parser.parse_args()


def _image_id(path: Path, source_root: Path) -> str:
    if source_root.is_dir():
        return path.relative_to(source_root).as_posix()
    return path.name


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    images = list_images(source_root)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {source_root}")

    detector = UltralyticsDetector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    config = InferenceConfig(
        preview_imgsz=args.preview_imgsz,
        fixed_slice_size=args.fixed_slice_size,
        fixed_overlap=args.fixed_overlap,
        merge_iou_threshold=args.merge_iou,
        postprocess=args.postprocess,
    )

    predictions: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for index, image_path in enumerate(images, start=1):
        image = read_image(image_path)
        result = run_image_inference(image, detector, mode=args.mode, config=config)
        image_id = _image_id(image_path, source_root)
        predictions[image_id] = result.detections
        metadata[image_id] = result.metadata | {"source_path": str(image_path)}
        print(
            f"[{index}/{len(images)}] {image_id}: "
            f"{len(result.detections)} boxes, {result.metadata['latency_ms']:.1f} ms, "
            f"slices={result.metadata.get('slice_count', 0)}"
        )

    save_prediction_jsonl(args.output, predictions, per_image_metadata=metadata)
    summary = _summarize(args, images, metadata)
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Predictions: {args.output}")
    print(f"Summary: {summary_path}")


def _summarize(args: argparse.Namespace, images: list[Path], metadata: dict[str, dict]) -> dict:
    latencies = [item.get("latency_ms", 0.0) for item in metadata.values()]
    slices = [item.get("slice_count", 0.0) for item in metadata.values()]
    detections = [item.get("detections", 0.0) for item in metadata.values()]
    return {
        "mode": args.mode,
        "weights": args.weights,
        "source": args.source,
        "num_images": len(images),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "avg_slice_count": float(np.mean(slices)) if slices else 0.0,
        "avg_detections": float(np.mean(detections)) if detections else 0.0,
        "config": {
            "conf": args.conf,
            "detector_iou": args.iou,
            "preview_imgsz": args.preview_imgsz,
            "fixed_slice_size": args.fixed_slice_size,
            "fixed_overlap": args.fixed_overlap,
            "merge_iou": args.merge_iou,
            "postprocess": args.postprocess,
        },
    }


if __name__ == "__main__":
    main()
