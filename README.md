# ASG-SAHI: Adaptive SAHI-TTA for UAV Small Object Detection

This workspace implements a minimal research codebase for the paper plan:
adaptive scale-guided sliced inference plus lite test-time augmentation for
small object detection on VisDrone-style UAV imagery.

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
  --output-root /kaggle/working/VisDronePrepared
```

This creates:

- `/kaggle/working/VisDronePrepared/images/train`
- `/kaggle/working/VisDronePrepared/images/val`
- `/kaggle/working/VisDronePrepared/labels/train`
- `/kaggle/working/VisDronePrepared/labels/val`
- `/kaggle/working/VisDronePrepared/VisDrone-prepared.yaml`

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
python make_figures.py --metrics-csv runs/metrics/all_methods.csv --output-dir figures
```

## Verification

```bash
python -m pytest -q
```

On this Windows workspace, prefer `python -m pytest` over the bare `pytest`
entrypoint so the repository root is on `sys.path`.
