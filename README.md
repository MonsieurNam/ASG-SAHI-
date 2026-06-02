# STD-YOLO / ASG-SAHI Research Codebase

This workspace implements a compact VisDrone research codebase for two related
small-object detection tracks:

- ASG-SAHI inference ablations for full-image, sliced, adaptive sliced, TTA, and WBF modes.
- STD-YOLO sliced-teacher distillation, where fixed SAHI mines pseudo labels for
  training a faster single-pass student detector.

## Environment

```bash
pip install -r requirements.txt
```

The core package is testable without a GPU. Training and inference require
`ultralytics` and a CUDA runtime for practical speed.

## Kaggle VisDrone Dataset Preparation

Your Kaggle input path is read-only, so prepare labels and YAML under
`/kaggle/working` before training:

```bash
python prepare_visdrone.py \
  --source-root /kaggle/input/datasets/kushagrapandya/visdrone-dataset \
  --output-root /kaggle/working/VisDronePrepared \
  --image-mode symlink
```

This creates:

- `/kaggle/working/VisDronePrepared/images/train`
- `/kaggle/working/VisDronePrepared/images/val`
- `/kaggle/working/VisDronePrepared/labels/train`
- `/kaggle/working/VisDronePrepared/labels/val`
- `/kaggle/working/VisDronePrepared/VisDrone-prepared.yaml`

If the cell appears to run silently, use the current script version: it prints
progress every 500 images. A successful run should show lines like
`[prepare:train] 500/6471 images`. If symlinks are unavailable, the script
falls back to copying and prints that fallback.

## Day-1 Baseline Training

```bash
python train_yolo.py \
  --data /kaggle/working/VisDronePrepared/VisDrone-prepared.yaml \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Fallback for a tight T4 budget:

```bash
python train_yolo.py \
  --data /kaggle/working/VisDronePrepared/VisDrone-prepared.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

## Day-2 to Day-4 Inference Matrix

```bash
python run_experiment.py --mode full --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/full.jsonl --device 0

python run_experiment.py --mode tta --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/tta.jsonl --device 0

python run_experiment.py --mode fixed_sahi --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/fixed_sahi.jsonl --device 0

python run_experiment.py --mode fixed_sahi_wbf --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/fixed_sahi_wbf.jsonl --device 0

python run_experiment.py --mode asg_sahi --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/asg_sahi.jsonl --device 0

python run_experiment.py --mode asg_sahi_tta --weights runs/train/exp/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val --output runs/preds/asg_sahi_tta.jsonl --device 0
```

Use `--limit 5` for smoke tests before full validation.

## Evaluation

```bash
python evaluate_results.py \
  --predictions runs/preds/asg_sahi_tta.jsonl \
  --images-dir /kaggle/working/VisDronePrepared/images/val \
  --labels-dir /kaggle/working/VisDronePrepared/labels/val \
  --method asg_sahi_tta \
  --output-prefix runs/metrics/asg_sahi_tta
```

Merge the `*.summary.csv` files into one CSV, then generate figures:

```bash
python make_figures.py --metrics-dir runs/metrics --output-dir runs/figures
```

## STD-YOLO Sliced-Teacher Distillation

Generate fixed-SAHI teacher predictions on the train split:

```bash
python run_experiment.py \
  --mode fixed_sahi \
  --weights runs/detect/runs/train/visdrone_yolov8s_20260601_164833/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/train \
  --output runs/preds/fixed_sahi_train.jsonl \
  --summary runs/preds/fixed_sahi_train.summary.json \
  --device 0 \
  --postprocess nms
```

Build the distilled dataset:

```bash
python build_distilled_dataset.py \
  --prepared-root /kaggle/working/VisDronePrepared \
  --teacher-preds runs/preds/fixed_sahi_train.jsonl \
  --output-root /kaggle/working/VisDroneSTD-small030 \
  --classes 0,1,2,6,7,9 \
  --min-conf 0.30 \
  --max-area-ratio 0.025 \
  --same-class-iou-drop 0.30 \
  --cross-class-iou-drop 0.40 \
  --max-pseudo-per-image 80 \
  --image-mode symlink
```

Analyze pseudo-label quality:

```bash
python analyze_pseudolabels.py \
  --manifest /kaggle/working/VisDroneSTD-small030/pseudo_manifest.jsonl \
  --output-prefix runs/analysis/std_small030
```

Train and evaluate the single-pass student:

```bash
python train_yolo.py \
  --data /kaggle/working/VisDroneSTD-small030/VisDrone-std.yaml \
  --model runs/detect/runs/train/visdrone_yolov8s_20260601_164833/weights/best.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name visdrone_std_yolov8s_small030

python run_experiment.py \
  --mode full \
  --weights runs/detect/runs/train/visdrone_std_yolov8s_small030/weights/best.pt \
  --source /kaggle/working/VisDronePrepared/images/val \
  --output runs/preds/std_yolo_full.jsonl \
  --summary runs/preds/std_yolo_full.summary.json \
  --device 0

python evaluate_results.py \
  --predictions runs/preds/std_yolo_full.jsonl \
  --images-dir /kaggle/working/VisDronePrepared/images/val \
  --labels-dir /kaggle/working/VisDronePrepared/labels/val \
  --method std_yolo_full \
  --output-prefix runs/metrics/std_yolo_full
```

## Verification

```bash
python -m pytest -q
```

On this Windows workspace, prefer `python -m pytest` over the bare `pytest`
entrypoint so the repository root is on `sys.path`.
