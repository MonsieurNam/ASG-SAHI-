"""Train a YOLO detector for VisDrone/YOLO-format datasets.

Example:
    python train_yolo.py --data VisDrone.yaml --model yolov8s.pt --epochs 80 --imgsz 640 --batch 16
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO for ASG-SAHI experiments.")
    parser.add_argument("--data", default="VisDrone.yaml", help="Ultralytics dataset YAML path/name.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO weights or model YAML.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Leave unset for auto.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from exc

    run_name = args.name or f"visdrone_{Path(args.model).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / run_name))
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"resolved_run_name": run_name, "save_dir": str(save_dir)}
    (save_dir / "train_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Training artifacts: {save_dir}")
    print(f"Best weights expected at: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
