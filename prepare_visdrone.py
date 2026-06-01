"""Prepare raw VisDrone DET folders for Ultralytics YOLO training.

Kaggle datasets are mounted read-only under ``/kaggle/input``. This script
converts VisDrone annotation text files to YOLO labels under a writable output
directory, then writes a dataset YAML for ``train_yolo.py``.

Example on Kaggle:
    python prepare_visdrone.py \
      --source-root /kaggle/input/datasets/kushagrapandya/visdrone-dataset \
      --output-root /kaggle/working/VisDronePrepared
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw VisDrone DET annotations to YOLO format.")
    parser.add_argument(
        "--source-root",
        default="/kaggle/input/datasets/kushagrapandya/visdrone-dataset",
        help="Folder containing VisDrone2019-DET-* directories.",
    )
    parser.add_argument("--output-root", default="/kaggle/working/VisDronePrepared")
    parser.add_argument("--no-copy-images", action="store_true", help="Write labels/YAML only; do not copy images.")
    parser.add_argument("--include-test-dev", action="store_true", help="Convert test-dev images/labels if annotations exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    requested_splits = ["train", "val"]
    if args.include_test_dev:
        requested_splits.append("test-dev")

    for split in requested_splits:
        source_split = source_root / SPLIT_DIRS[split]
        if not source_split.exists():
            raise SystemExit(f"Missing VisDrone split folder: {source_split}")
        summaries[split] = convert_visdrone_split(
            source_split=source_split,
            output_root=output_root,
            split_name=split,
            copy_images=not args.no_copy_images,
        )

    yaml_path = write_dataset_yaml(output_root, output_root / "VisDrone-prepared.yaml")
    summary_path = output_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Prepared dataset YAML: {yaml_path}")
    print(f"Summary: {summary_path}")
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary['images']} boxes={summary['boxes']} "
            f"ignored={summary['ignored']} empty_labels={summary['empty_labels']}"
        )


def convert_visdrone_split(
    source_split: Path | str,
    output_root: Path | str,
    split_name: str,
    copy_images: bool = True,
) -> dict[str, int]:
    """Convert one VisDrone DET split to YOLO label files.

    VisDrone rows are:
    ``bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion``.
    Object category 0 and score 0 are ignored regions. Categories 1..10 map to
    YOLO classes 0..9.
    """

    source = Path(source_split)
    output = Path(output_root)
    image_src, ann_src = resolve_split_dirs(source)
    image_dst = output / "images" / split_name
    label_dst = output / "labels" / split_name
    image_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in image_src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_src}")

    summary = {"images": 0, "boxes": 0, "ignored": 0, "empty_labels": 0}
    for image_path in image_paths:
        width, height = _read_image_size(image_path)
        annotation_path = ann_src / f"{image_path.stem}.txt"
        yolo_rows: list[str] = []
        ignored = 0

        if annotation_path.exists():
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                converted = _convert_annotation_row(line, image_width=width, image_height=height)
                if converted is None:
                    ignored += 1
                    continue
                yolo_rows.append(converted)

        (label_dst / f"{image_path.stem}.txt").write_text("\n".join(yolo_rows), encoding="utf-8")
        if copy_images:
            shutil.copy2(image_path, image_dst / image_path.name)

        summary["images"] += 1
        summary["boxes"] += len(yolo_rows)
        summary["ignored"] += ignored
        if not yolo_rows:
            summary["empty_labels"] += 1

    return summary


def resolve_split_dirs(source_split: Path | str) -> tuple[Path, Path]:
    """Resolve VisDrone image/annotation dirs across common Kaggle layouts."""

    source = Path(source_split)
    candidates = [
        source,
        source / source.name,
    ]
    candidates.extend(path for path in source.iterdir() if path.is_dir()) if source.exists() else None

    for base in candidates:
        image_dir = _first_existing_dir(base, ["images", "Images", "JPEGImages"])
        if image_dir is not None:
            annotation_dir = _first_existing_dir(base, ["annotations", "Annotations", "labels"])
            return image_dir, annotation_dir or (base / "annotations")

        direct_images = _has_images(base)
        if direct_images:
            annotation_dir = _first_existing_dir(base, ["annotations", "Annotations", "labels"])
            return base, annotation_dir or (base / "annotations")

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find VisDrone images directory. Checked direct, nested, and child folders under: "
        f"{checked}"
    )


def _first_existing_dir(base: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _has_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.suffix.lower() in {".jpg", ".jpeg", ".png"} for child in path.iterdir())


def write_dataset_yaml(output_root: Path | str, yaml_path: Path | str) -> Path:
    output = Path(output_root)
    yaml_file = Path(yaml_path)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {idx}: {name}" for idx, name in VISDRONE_NAMES.items())
    yaml_file.write_text(
        "\n".join(
            [
                f"path: {output}",
                "train: images/train",
                "val: images/val",
                "test: images/test-dev",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_file


def _convert_annotation_row(line: str, image_width: int, image_height: int) -> str | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        raise ValueError(f"invalid VisDrone annotation row: {line!r}")

    x, y, width, height = map(float, parts[:4])
    score = int(float(parts[4]))
    category = int(float(parts[5]))
    if score == 0 or category <= 0:
        return None

    class_id = category - 1
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    x_center, y_center, norm_width, norm_height = [
        min(1.0, max(0.0, value)) for value in (x_center, y_center, norm_width, norm_height)
    ]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ModuleNotFoundError:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        height, width = image.shape[:2]
        return width, height


if __name__ == "__main__":
    main()
