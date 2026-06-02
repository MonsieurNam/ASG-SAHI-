"""Build a sliced-teacher distilled VisDrone dataset for YOLO training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from adaptive_sahi.io import image_size, list_images, load_prediction_jsonl, load_yolo_labels_xyxy
from adaptive_sahi.pseudolabels import (
    DROP_KEYS,
    PseudoLabelConfig,
    filter_pseudo_labels,
    parse_class_ids,
    summary_total,
    xyxy_to_yolo_row,
)
from prepare_visdrone import VISDRONE_NAMES, materialize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VisDrone dataset augmented with sliced-teacher pseudo labels.")
    parser.add_argument("--prepared-root", required=True, help="Prepared VisDrone root with images/ and labels/.")
    parser.add_argument("--teacher-preds", required=True, help="Teacher prediction JSONL from run_experiment.py.")
    parser.add_argument("--output-root", required=True, help="Output dataset root.")
    parser.add_argument("--classes", default="0,1,2,6,7,9", help="Comma-separated class IDs allowed for pseudo labels.")
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--max-area-ratio", type=float, default=0.025)
    parser.add_argument("--same-class-iou-drop", type=float, default=0.30)
    parser.add_argument("--cross-class-iou-drop", type=float, default=0.40)
    parser.add_argument("--max-pseudo-per-image", type=int, default=80)
    parser.add_argument("--image-mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PseudoLabelConfig(
        class_ids=parse_class_ids(args.classes),
        min_conf=args.min_conf,
        max_area_ratio=args.max_area_ratio,
        same_class_iou_drop=args.same_class_iou_drop,
        cross_class_iou_drop=args.cross_class_iou_drop,
        max_pseudo_per_image=args.max_pseudo_per_image,
    )
    summary = build_distilled_dataset(
        prepared_root=Path(args.prepared_root),
        teacher_preds=Path(args.teacher_preds),
        output_root=Path(args.output_root),
        config=config,
        image_mode=args.image_mode,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_distilled_dataset(
    prepared_root: Path | str,
    teacher_preds: Path | str,
    output_root: Path | str,
    config: PseudoLabelConfig,
    image_mode: str = "symlink",
    limit: int | None = None,
) -> dict:
    """Create a distilled dataset and return aggregate pseudo-label statistics."""

    prepared = Path(prepared_root)
    output = Path(output_root)
    teacher_predictions, _ = load_prediction_jsonl(teacher_preds)
    _ensure_split_dirs(output)

    manifest_path = output / "pseudo_manifest.jsonl"
    summary = _empty_summary(config)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        train_summary = _build_train_split(
            prepared=prepared,
            output=output,
            teacher_predictions=teacher_predictions,
            config=config,
            image_mode=image_mode,
            manifest=manifest,
            limit=limit,
        )
        summary.update(train_summary)

    val_summary = _copy_split_unchanged(prepared, output, "val", image_mode=image_mode, limit=limit)
    summary["val_images"] = val_summary["images"]
    summary["val_labels_copied"] = val_summary["labels"]

    yaml_path = write_distilled_yaml(output, output / "VisDrone-std.yaml")
    summary["yaml"] = str(yaml_path)
    summary["manifest"] = str(manifest_path)
    (output / "distill_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_distilled_yaml(output_root: Path | str, yaml_path: Path | str) -> Path:
    output = Path(output_root)
    yaml_file = Path(yaml_path)
    names = "\n".join(f"  {idx}: {name}" for idx, name in VISDRONE_NAMES.items())
    yaml_file.write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_file


def _build_train_split(
    prepared: Path,
    output: Path,
    teacher_predictions: dict[str, np.ndarray],
    config: PseudoLabelConfig,
    image_mode: str,
    manifest,
    limit: int | None,
) -> dict:
    image_root = prepared / "images" / "train"
    label_root = prepared / "labels" / "train"
    images = list_images(image_root)
    if limit is not None:
        images = images[:limit]

    aggregate = {
        "train_images": 0,
        "original_gt": 0,
        "teacher_candidates": 0,
        "kept_pseudo": 0,
        "dropped_total": 0,
        **{key: 0 for key in DROP_KEYS},
    }
    for index, image_path in enumerate(images, start=1):
        image_id = image_path.relative_to(image_root).as_posix()
        width, height = image_size(image_path)
        label_path = label_root / Path(image_id).with_suffix(".txt")
        gt = load_yolo_labels_xyxy(label_path, width, height)
        teacher = _lookup_teacher_predictions(teacher_predictions, image_id)
        kept, records, per_image = filter_pseudo_labels(
            teacher,
            gt,
            image_width=width,
            image_height=height,
            image_id=image_id,
            config=config,
        )

        original_rows = _read_label_rows(label_path)
        pseudo_rows = [xyxy_to_yolo_row(row, width, height) for row in kept]
        _write_label_rows(output / "labels" / "train" / Path(image_id).with_suffix(".txt"), original_rows + pseudo_rows)
        materialize_image(image_path, output / "images" / "train" / image_id, image_mode=image_mode)

        record = {
            "image_id": image_id,
            "original_gt": int(len(gt)),
            "teacher_candidates": int(len(teacher)),
            "kept": records,
            "summary": per_image,
        }
        manifest.write(json.dumps(record, sort_keys=True) + "\n")

        aggregate["train_images"] += 1
        aggregate["original_gt"] += int(len(gt))
        aggregate["teacher_candidates"] += int(len(teacher))
        aggregate["kept_pseudo"] += int(len(kept))
        aggregate["dropped_total"] += summary_total(per_image)
        for key in DROP_KEYS:
            aggregate[key] += int(per_image.get(key, 0))

        if index == 1 or index % 500 == 0 or index == len(images):
            print(
                f"[distill:train] {index}/{len(images)} images, "
                f"kept_pseudo={aggregate['kept_pseudo']}, dropped={aggregate['dropped_total']}",
                flush=True,
            )
    return aggregate


def _copy_split_unchanged(
    prepared: Path,
    output: Path,
    split: str,
    image_mode: str,
    limit: int | None,
) -> dict[str, int]:
    image_root = prepared / "images" / split
    label_root = prepared / "labels" / split
    images = list_images(image_root)
    if limit is not None:
        images = images[:limit]

    summary = {"images": 0, "labels": 0}
    for image_path in images:
        image_id = image_path.relative_to(image_root).as_posix()
        label_path = label_root / Path(image_id).with_suffix(".txt")
        materialize_image(image_path, output / "images" / split / image_id, image_mode=image_mode)
        dst_label = output / "labels" / split / Path(image_id).with_suffix(".txt")
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        if label_path.exists():
            shutil.copy2(label_path, dst_label)
            summary["labels"] += 1
        else:
            dst_label.write_text("", encoding="utf-8")
        summary["images"] += 1
    return summary


def _lookup_teacher_predictions(predictions: dict[str, np.ndarray], image_id: str) -> np.ndarray:
    if image_id in predictions:
        return predictions[image_id]
    name = Path(image_id).name
    if name in predictions:
        return predictions[name]
    return np.zeros((0, 6), dtype=float)


def _read_label_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_label_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _ensure_split_dirs(output: Path) -> None:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _empty_summary(config: PseudoLabelConfig) -> dict:
    return {
        "classes": sorted(config.class_ids),
        "min_conf": config.min_conf,
        "max_area_ratio": config.max_area_ratio,
        "same_class_iou_drop": config.same_class_iou_drop,
        "cross_class_iou_drop": config.cross_class_iou_drop,
        "max_pseudo_per_image": config.max_pseudo_per_image,
    }


if __name__ == "__main__":
    main()
