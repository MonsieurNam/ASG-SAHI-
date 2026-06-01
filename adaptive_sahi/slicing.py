"""Image slicing utilities for tiled inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SliceWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]

    starts = list(range(0, length - window + 1, step))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_slices(
    image_width: int,
    image_height: int,
    slice_width: int,
    slice_height: int | None = None,
    overlap: float = 0.25,
) -> list[SliceWindow]:
    """Generate edge-covering slice windows.

    ``overlap`` is a fraction in ``[0, 1)``. Edge slices are shifted backward so
    the final window touches the right/bottom border without exceeding slice size.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if slice_width <= 0:
        raise ValueError("slice_width must be positive")
    if slice_height is None:
        slice_height = slice_width
    if slice_height <= 0:
        raise ValueError("slice_height must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    step_x = max(1, int(round(slice_width * (1.0 - overlap))))
    step_y = max(1, int(round(slice_height * (1.0 - overlap))))
    x_starts = _axis_starts(image_width, slice_width, step_x)
    y_starts = _axis_starts(image_height, slice_height, step_y)

    windows: list[SliceWindow] = []
    for y in y_starts:
        for x in x_starts:
            windows.append(
                SliceWindow(
                    x1=x,
                    y1=y,
                    x2=min(x + slice_width, image_width),
                    y2=min(y + slice_height, image_height),
                )
            )
    return windows
