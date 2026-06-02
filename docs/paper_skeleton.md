# STD-YOLO Paper Skeleton

## Tentative Title

STD-YOLO: Transferring Sliced-Teacher Knowledge into Single-Pass UAV Small Object Detection

## Abstract Draft

Small object detection in UAV imagery remains difficult because distant targets occupy only a few pixels and often appear in dense, occluded scenes. Sliced inference such as SAHI improves recall by evaluating local crops at higher effective resolution, but this accuracy gain comes with higher inference latency and redundant computation. This paper proposes STD-YOLO, a sliced-teacher distillation pipeline that uses a SAHI-enhanced detector as a teacher to mine high-confidence small-object pseudo labels on the training split, then fine-tunes a standard YOLO student for single-pass inference. The method filters teacher detections by class, confidence, object scale, and overlap with ground-truth annotations to reduce pseudo-label noise. Experiments on VisDrone2019-DET compare full-image YOLOv8s, fixed SAHI teacher inference, and the distilled single-pass student using mAP, class-wise AP, latency, and pseudo-label quality statistics. The goal is to recover part of the sliced-inference accuracy gain while retaining near full-image deployment speed.

## 1. Introduction

- UAV detection has a practical accuracy-latency conflict: full-image detectors are fast but miss tiny objects; sliced inference detects more tiny objects but is slower.
- Existing adaptive slicing methods reduce redundancy, but still require tiled inference at deployment.
- STD-YOLO changes the deployment point: slicing is used only as an offline teacher during training.
- Contributions:
  - A sliced-teacher pseudo-labeling pipeline for VisDrone small classes.
  - Confidence, scale, and GT-overlap filters to control pseudo-label noise.
  - A single-pass student comparison against both full-image YOLO and fixed SAHI teacher.
  - Accuracy-latency and class-wise analysis focused on small UAV targets.

## 2. Related Work

- UAV small object detection with YOLO-family detectors.
- Sliced inference and fine-tuning: SAHI, ASAHI, density-guided variants.
- Knowledge distillation and pseudo-labeling for aerial object detection.
- VisDrone2019-DET as a dense UAV benchmark.

## 3. Method

### 3.1 Baseline and Teacher

Train YOLOv8s on VisDrone at image size 640. Use the trained detector in two inference modes:

- full-image baseline;
- fixed SAHI teacher with 640-pixel slices and 0.25 overlap.

The fixed SAHI teacher provides an upper-bound accuracy signal but is not the deployment method.

### 3.2 Sliced-Teacher Pseudo-Label Mining

Run fixed SAHI on the training images and keep only pseudo detections satisfying:

- class in pedestrian, people, bicycle, tricycle, awning-tricycle, motor;
- confidence at least 0.30;
- box area ratio at most 0.025;
- IoU with same-class GT at most 0.30;
- IoU with cross-class GT at most 0.40;
- at most 80 pseudo labels per image.

The final train labels are the union of original GT and filtered pseudo labels. Validation labels remain unchanged.

### 3.3 Student Training

Fine-tune YOLOv8s from the baseline checkpoint on the distilled dataset. At evaluation time, the student uses full-image single-pass inference only.

## 4. Experiments

### Dataset

VisDrone2019-DET train/val prepared in YOLO format.

### Compared Methods

1. YOLOv8s full-image baseline.
2. YOLOv8s + fixed SAHI teacher.
3. STD-YOLO full-image student.
4. Optional control: YOLOv8s fine-tuned for the same number of epochs on original labels only.

### Metrics

- mAP@0.50 and mAP@0.50:0.95.
- Class-wise AP for small classes.
- Average latency per image.
- Pseudo-label counts, score distribution, and drop reasons.

## 5. Results and Discussion

### Main Result Table

| Method | mAP50 | mAP50-95 | Avg latency | Deployment |
|---|---:|---:|---:|---|
| full | 0.3635 | 0.2128 | 29.0 ms | single-pass |
| fixed_sahi | 0.4203 | 0.2429 | 96.6 ms | sliced |
| std_yolo_full | TBD | TBD | TBD | single-pass |
| ft_original50 | optional | optional | optional | single-pass |

### Expected Discussion Angles

- Fixed SAHI confirms that sliced teacher knowledge exists, especially for pedestrian, people, bicycle, and motor.
- STD-YOLO is successful if it improves full-image mAP while keeping latency near the original detector.
- If fine-tuning control also improves, the discussion must separate generic extra training from pseudo-label distillation.
- Pseudo-label noise remains the key limitation.

## 6. Limitations

- The method depends on teacher quality.
- Pseudo labels can reinforce teacher false positives.
- Main evaluation is VisDrone-only.
- The current implementation transfers box labels, not feature-level distillation.

## 7. Conclusion

STD-YOLO uses sliced inference as an offline teacher rather than a deployment-time requirement, targeting a practical balance between small-object recall and single-pass inference speed for UAV imagery.
