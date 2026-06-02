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
import os
import shutil
import sys
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
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How to expose images in output. symlink is fastest on Kaggle; copy is safest; none writes labels/YAML only.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Deprecated alias for --image-mode none.",
    )
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N images.")
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
    image_mode = "none" if args.no_copy_images else args.image_mode

    for split in requested_splits:
        source_split = source_root / SPLIT_DIRS[split]
        if not source_split.exists():
            raise SystemExit(f"Missing VisDrone split folder: {source_split}")
        summaries[split] = convert_visdrone_split(
            source_split=source_split,
            output_root=output_root,
            split_name=split,
            image_mode=image_mode,
            progress_every=args.progress_every,
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
    image_mode: str | None = None,
    progress_every: int = 500,
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

    if image_mode is None:
        image_mode = "copy" if copy_images else "none"
    if image_mode not in {"symlink", "copy", "none"}:
        raise ValueError("image_mode must be one of: symlink, copy, none")

    print(
        f"[prepare:{split_name}] images={len(image_paths)} image_dir={image_src} "
        f"annotations={ann_src} image_mode={image_mode}",
        flush=True,
    )
    summary = {"images": 0, "boxes": 0, "ignored": 0, "empty_labels": 0, "linked": 0, "copied": 0}
    for index, image_path in enumerate(image_paths, start=1):
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
        used_mode = materialize_image(image_path, image_dst / image_path.name, image_mode=image_mode)
        if used_mode == "symlink":
            summary["linked"] += 1
        elif used_mode == "copy":
            summary["copied"] += 1

        summary["images"] += 1
        summary["boxes"] += len(yolo_rows)
        summary["ignored"] += ignored
        if not yolo_rows:
            summary["empty_labels"] += 1
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(image_paths)):
            print(
                f"[prepare:{split_name}] {index}/{len(image_paths)} images, "
                f"boxes={summary['boxes']}, ignored={summary['ignored']}, "
                f"linked={summary['linked']}, copied={summary['copied']}",
                flush=True,
            )

    return summary


def materialize_image(source_path: Path | str, output_path: Path | str, image_mode: str) -> str:
    """Expose one image in the prepared dataset and return the mode actually used."""

    src = Path(source_path)
    dst = Path(output_path)
    if image_mode == "none":
        return "none"
    if dst.exists() or dst.is_symlink():
        return "symlink" if dst.is_symlink() else "copy"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if image_mode == "symlink":
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError as exc:
            print(
                f"[prepare] symlink failed for {src.name}: {exc}. Falling back to copy.",
                file=sys.stderr,
                flush=True,
            )
    shutil.copy2(src, dst)
    return "copy"


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
