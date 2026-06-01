import numpy as np

from adaptive_sahi.geometry import (
    clip_boxes,
    remap_boxes_from_slice,
    unflip_boxes_horizontal,
)
from adaptive_sahi.slicing import generate_slices


def test_slice_to_global_roundtrip_with_clipping():
    local = np.array(
        [
            [0.0, 0.0, 50.0, 60.0, 0.9, 1],
            [-10.0, 20.0, 120.0, 90.0, 0.8, 2],
        ],
        dtype=float,
    )

    global_boxes = remap_boxes_from_slice(local, x_offset=100, y_offset=200, image_width=180, image_height=260)

    expected = np.array(
        [
            [100.0, 200.0, 150.0, 260.0, 0.9, 1],
            [90.0, 220.0, 180.0, 260.0, 0.8, 2],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(global_boxes, expected)


def test_horizontal_unflip_restores_original_coordinates():
    flipped_prediction = np.array([[10.0, 5.0, 30.0, 25.0, 0.7, 0]], dtype=float)

    restored = unflip_boxes_horizontal(flipped_prediction, image_width=100)

    expected = np.array([[70.0, 5.0, 90.0, 25.0, 0.7, 0]], dtype=float)
    np.testing.assert_allclose(restored, expected)


def test_clip_boxes_handles_empty_predictions():
    empty = np.zeros((0, 6), dtype=float)

    clipped = clip_boxes(empty, image_width=640, image_height=480)

    assert clipped.shape == (0, 6)


def test_generate_slices_covers_image_edges_with_overlap():
    slices = generate_slices(image_width=1000, image_height=700, slice_width=400, slice_height=400, overlap=0.25)

    assert slices[0].x1 == 0
    assert slices[0].y1 == 0
    assert any(s.x2 == 1000 for s in slices)
    assert any(s.y2 == 700 for s in slices)
    assert all((s.x2 - s.x1) <= 400 and (s.y2 - s.y1) <= 400 for s in slices)
