"""Generate paper figures from ASG-SAHI metrics and prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from adaptive_sahi.io import list_images, load_prediction_jsonl, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ASG-SAHI result figures.")
    parser.add_argument("--metrics-csv", default=None, help="CSV with method, mAP50, avg_latency_ms columns.")
    parser.add_argument("--metrics-dir", default=None, help="Directory containing *.summary.csv metric files.")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(args)
    metrics.to_csv(output_dir / "combined_metrics.csv", index=False)

    _plot_accuracy_latency(metrics, output_dir / "accuracy_latency.png")
    if "avg_slice_count" in metrics.columns:
        _plot_slice_ablation(metrics, output_dir / "slice_ablation.png")

    if args.images_dir and args.predictions_jsonl:
        _plot_qualitative_grid(
            images_dir=Path(args.images_dir),
            predictions_path=Path(args.predictions_jsonl),
            output_path=output_dir / "qualitative_grid.png",
            max_images=args.max_images,
            score_threshold=args.score_threshold,
        )


def _load_metrics(args: argparse.Namespace) -> pd.DataFrame:
    if args.metrics_csv:
        return pd.read_csv(args.metrics_csv)
    if args.metrics_dir:
        summary_paths = sorted(Path(args.metrics_dir).glob("*.summary.csv"))
        if not summary_paths:
            raise FileNotFoundError(f"No *.summary.csv files found in {args.metrics_dir}")
        frames = [pd.read_csv(path) for path in summary_paths]
        metrics = pd.concat(frames, ignore_index=True)
        if "method" in metrics.columns:
            metrics = metrics[metrics["method"] != "full_smoke"]
        return metrics
    raise SystemExit("Provide either --metrics-csv or --metrics-dir")


def _plot_accuracy_latency(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(metrics["avg_latency_ms"], metrics["mAP50"], s=80)
    for _, row in metrics.iterrows():
        ax.annotate(str(row["method"]), (row["avg_latency_ms"], row["mAP50"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Average latency (ms/image)")
    ax.set_ylabel("mAP@0.50")
    ax.set_title("Accuracy-latency trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_slice_ablation(metrics: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics.sort_values("avg_slice_count")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(ordered["method"], ordered["avg_slice_count"], color="#6aa6d9")
    ax1.set_ylabel("Average slices/image")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(ordered["method"], ordered["mAP50"], color="#d95f02", marker="o")
    ax2.set_ylabel("mAP@0.50")
    ax1.set_title("Slice budget versus accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_qualitative_grid(
    images_dir: Path,
    predictions_path: Path,
    output_path: Path,
    max_images: int,
    score_threshold: float,
) -> None:
    predictions, _ = load_prediction_jsonl(predictions_path)
    image_index = {path.name: path for path in list_images(images_dir)}
    selected = list(predictions)[:max_images]
    if not selected:
        return

    cols = min(3, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for ax in [item for row in axes for item in row]:
        ax.axis("off")

    for idx, image_id in enumerate(selected):
        image_path = images_dir / image_id
        if not image_path.exists():
            image_path = image_index.get(Path(image_id).name)
        if image_path is None:
            continue
        image = read_image(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = predictions[image_id]
        for x1, y1, x2, y2, score, class_id in boxes:
            if score < score_threshold:
                continue
            cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 40), 2)
            cv2.putText(
                rgb,
                f"{int(class_id)}:{score:.2f}",
                (int(x1), max(15, int(y1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 80, 40),
                1,
                cv2.LINE_AA,
            )
        row, col = divmod(idx, cols)
        axes[row][col].imshow(rgb)
        axes[row][col].set_title(image_id)
        axes[row][col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
