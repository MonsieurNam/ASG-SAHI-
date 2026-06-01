import numpy as np

from adaptive_sahi.engine import InferenceConfig, run_image_inference


class ConstantDetector:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=float)
        self.calls = []

    def predict(self, image, imgsz=None, augment=False):
        self.calls.append({"shape": image.shape, "imgsz": imgsz, "augment": augment})
        return self.prediction.copy()


class HighDensityPreviewDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, image, imgsz=None, augment=False):
        self.calls += 1
        if self.calls == 1:
            rows = []
            for idx in range(40):
                x = float((idx % 10) * 20)
                y = float((idx // 10) * 20)
                rows.append([x, y, x + 8, y + 8, 0.5, 0])
            return np.asarray(rows, dtype=float)
        return np.array([[0.0, 0.0, 20.0, 20.0, 0.8, 0]], dtype=float)


def test_fixed_sahi_remaps_slice_predictions_and_records_slice_count():
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    detector = ConstantDetector([[0.0, 0.0, 20.0, 20.0, 0.8, 0]])

    result = run_image_inference(
        image,
        detector,
        mode="fixed_sahi",
        config=InferenceConfig(fixed_slice_size=512, fixed_overlap=0.0, postprocess="none"),
    )

    assert result.metadata["slice_count"] == 4
    assert result.detections.shape == (4, 6)
    assert result.detections[:, 0].max() == 288.0
    assert result.detections[:, 1].max() == 288.0


def test_asg_sahi_tta_uses_high_density_policy_and_keeps_boxes_in_bounds():
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    detector = HighDensityPreviewDetector()

    result = run_image_inference(
        image,
        detector,
        mode="asg_sahi_tta",
        config=InferenceConfig(postprocess="nms"),
    )

    assert result.metadata["policy_level"] == "high"
    assert result.metadata["hflip_tta_used"] is True
    assert result.metadata["slice_count"] > 0
    assert result.detections.shape[1] == 6
    assert np.all(result.detections[:, 0] >= 0)
    assert np.all(result.detections[:, 1] >= 0)
    assert np.all(result.detections[:, 2] <= 800)
    assert np.all(result.detections[:, 3] <= 800)
