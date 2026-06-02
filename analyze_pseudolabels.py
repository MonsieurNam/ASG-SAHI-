"""Summarize pseudo-label manifests produced by build_distilled_dataset.py."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_sahi.pseudolabels import DROP_KEYS
from prepare_visdrone import VISDRONE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sliced-teacher pseudo-label manifest quality.")
    parser.add_argument("--manifest", required=True, help="pseudo_manifest.jsonl from build_distilled_dataset.py")
    parser.add_argument("--output-prefix", required=True, help="Prefix for CSV/Markdown outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_manifest(Path(args.manifest), Path(args.output_prefix))
    print(json.dumps(summary, indent=2, sort_keys=True))


def analyze_manifest(manifest_path: Path | str, output_prefix: Path | str) -> dict:
    manifest = Path(manifest_path)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    image_count = 0
    total_gt = 0
    total_teacher = 0
    kept_total = 0
    class_counts: Counter[int] = Counter()
    drop_counts: Counter[str] = Counter()
    scores: list[float] = []
    area_ratios: list[float] = []
    per_image_kept: list[int] = []

    with manifest.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kept = record.get("kept", [])
            summary = record.get("summary", {})
            image_count += 1
            total_gt += int(record.get("original_gt", 0))
            total_teacher += int(record.get("teacher_candidates", 0))
            kept_total += len(kept)
            per_image_kept.append(len(kept))
            for key in DROP_KEYS:
                drop_counts[key] += int(summary.get(key, 0))
            for item in kept:
                class_id = int(item["class_id"])
                class_counts[class_id] += 1
                scores.append(float(item["score"]))
                area_ratios.append(float(item["area_ratio"]))

    summary_row = {
        "images": image_count,
        "original_gt": total_gt,
        "teacher_candidates": total_teacher,
        "kept_pseudo": kept_total,
        "avg_pseudo_per_image": _mean(per_image_kept),
        "avg_score": _mean(scores),
        "avg_area_ratio": _mean(area_ratios),
        **{key: drop_counts[key] for key in DROP_KEYS},
    }
    _write_summary_csv(prefix.with_suffix(".summary.csv"), summary_row)
    _write_class_csv(prefix.with_suffix(".per_class.csv"), class_counts)
    _write_hist_csv(prefix.with_suffix(".score_hist.csv"), scores, bins=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    _write_markdown(prefix.with_suffix(".summary.md"), summary_row, class_counts)
    return summary_row


def _write_summary_csv(path: Path, row: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_class_csv(path: Path, class_counts: Counter[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "pseudo_count"])
        writer.writeheader()
        for class_id in sorted(class_counts):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": VISDRONE_NAMES.get(class_id, str(class_id)),
                    "pseudo_count": class_counts[class_id],
                }
            )


def _write_hist_csv(path: Path, values: list[float], bins: list[float]) -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    for value in values:
        for low, high in zip(bins[:-1], bins[1:]):
            if low <= value < high:
                counts[f"{low:.2f}-{min(high, 1.0):.2f}"] += 1
                break
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["score_bin", "count"])
        writer.writeheader()
        for low, high in zip(bins[:-1], bins[1:]):
            label = f"{low:.2f}-{min(high, 1.0):.2f}"
            writer.writerow({"score_bin": label, "count": counts[label]})


def _write_markdown(path: Path, summary_row: dict, class_counts: Counter[int]) -> None:
    lines = [
        "# Pseudo-label Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary_row.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Per-class Pseudo Labels", "", "| Class | Count |", "|---|---:|"])
    for class_id in sorted(class_counts):
        lines.append(f"| {VISDRONE_NAMES.get(class_id, str(class_id))} | {class_counts[class_id]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
