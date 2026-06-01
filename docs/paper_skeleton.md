# ASG-SAHI Paper Skeleton

## Tentative Title

ASG-SAHI: Adaptive Scale-Guided Sliced Inference with Stability-Aware Test-Time Augmentation for Small Object Detection in UAV Imagery

## Abstract Draft

Small object detection in UAV imagery remains difficult because distant and densely packed targets occupy only a few pixels in high-resolution frames. Slicing-aided inference improves detector recall by magnifying local regions, but fixed slicing introduces redundant computation and can over-process sparse images. This paper proposes ASG-SAHI, an adaptive scale-guided inference pipeline that selects slice size and overlap from a low-resolution preview pass, then applies horizontal-flip test-time augmentation only for high-density scenes. Class-safe weighted box fusion merges detections from slices and augmented views. Experiments on VisDrone2019-DET compare full-image YOLOv8 inference, standard TTA, fixed SAHI, fixed SAHI with WBF, ASG-SAHI, and ASG-SAHI with lite TTA under the same detector weights. The study reports mAP, class-wise AP, latency, and average slices per image to evaluate the accuracy-efficiency trade-off on T4-level hardware.

## 1. Introduction

- UAV imagery contains tiny, dense, and occluded targets.
- Full-image YOLO inference is efficient but loses fine object detail.
- Fixed SAHI improves local resolution but increases redundant computation.
- TTA and WBF can recover unstable detections, yet full TTA is costly.
- Contributions:
  - Preview-density adaptive slicing policy.
  - High-density-only lite TTA.
  - Class-safe WBF/NMS merge layer.
  - Reproducible VisDrone accuracy-latency ablation.

## 2. Related Work

- YOLO-based small object detection: YOLOv8, SL-YOLO, MSD-YOLO.
- Sliced inference: SAHI, ASAHI, density/adaptive slicing.
- Test-time augmentation and box fusion: hflip/multiscale TTA, WBF.
- UAV detection benchmark: VisDrone2019-DET.

## 3. Method

### 3.1 Baseline Detector

Train YOLOv8s on the official VisDrone train split using image size 640 and seed 42. Use the same weights for all inference modes.

### 3.2 Fixed SAHI Baseline

Slice each image into 640-pixel windows with 0.25 overlap, remap local boxes to global coordinates, then apply class-safe NMS or WBF.

### 3.3 ASG-SAHI Policy

Run a preview detector pass and compute:

- boxes per megapixel;
- median predicted box area divided by image area.

Policy:

- low density: 768-pixel slices, 0.15 overlap, no TTA;
- medium density: 640-pixel slices, 0.25 overlap, no TTA;
- high density or tiny predicted scale: 512-pixel slices, 0.30 overlap, optional hflip TTA.

### 3.4 Post-Processing

Apply class-safe WBF for sliced/TTA modes. Use NMS as fallback and as a separate ablation.

## 4. Experiments

### Dataset

VisDrone2019-DET train/val. Test-dev is optional if labels/evaluation tooling are available.

### Compared Methods

1. YOLOv8s full-image inference.
2. YOLOv8s with Ultralytics augment/TTA.
3. YOLOv8s + fixed SAHI.
4. YOLOv8s + fixed SAHI + WBF.
5. ASG-SAHI.
6. ASG-SAHI + lite TTA.

### Metrics

- mAP@0.50;
- mAP@0.50:0.95;
- class-wise AP;
- average latency per image;
- average slices per image;
- qualitative failure cases.

## 5. Results and Discussion

Fill after running `evaluate_results.py`.

### Main Result Table

| Method | mAP50 | mAP50-95 | Avg latency | Avg slices |
|---|---:|---:|---:|---:|
| full | TBD | TBD | TBD | TBD |
| tta | TBD | TBD | TBD | TBD |
| fixed_sahi | TBD | TBD | TBD | TBD |
| fixed_sahi_wbf | TBD | TBD | TBD | TBD |
| asg_sahi | TBD | TBD | TBD | TBD |
| asg_sahi_tta | TBD | TBD | TBD | TBD |

### Expected Discussion Angles

- ASG-SAHI should reduce slice count relative to fixed SAHI on sparse images.
- Lite TTA should help high-density scenes more than low-density scenes.
- WBF may improve duplicate handling but can hurt if detector scores are poorly calibrated.

## 6. Limitations

- Single detector family.
- Main contribution is inference-time adaptation, not a new model architecture.
- Preview density depends on baseline detector quality.
- VisDrone-only result needs external validation later.

## 7. Conclusion

ASG-SAHI targets the practical accuracy-latency trade-off for UAV small object detection on limited GPU hardware.
