"""Turn jq_skin zone readings into a fixed-size tactile image.

The normalization range is fixed instead of being estimated per frame. This
preserves force magnitude: a light touch must not look identical to a hard
press merely because both are the largest value in their respective frame.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ZonePlacement:
    """Placement of one flattened tactile zone on the output canvas."""

    key: str
    grid_shape: tuple[int, int]
    top: int
    left: int
    height: int
    width: int
    flip_horizontal: bool = False

    def __post_init__(self) -> None:
        if len(self.grid_shape) != 2 or min(*self.grid_shape, self.height, self.width) <= 0:
            raise ValueError(f"Invalid tactile zone placement: {self}")


@dataclass(frozen=True)
class TactileLayout:
    """A concrete spatial layout for all tactile zones."""

    height: int
    width: int
    zones: tuple[ZonePlacement, ...]

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Tactile layout dimensions must be positive")
        for zone in self.zones:
            if zone.top < 0 or zone.left < 0:
                raise ValueError(f"Zone {zone.key!r} starts outside the canvas")
            if zone.top + zone.height > self.height or zone.left + zone.width > self.width:
                raise ValueError(f"Zone {zone.key!r} extends outside the canvas")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TactileLayout":
        """Load a pinned layout or an exported jq_skin layout document."""

        data = payload.get("layout", payload)
        zones = tuple(
            ZonePlacement(
                key=zone["key"],
                grid_shape=tuple(zone["grid_shape"]),
                top=zone["top"],
                left=zone["left"],
                height=zone["height"],
                width=zone["width"],
                flip_horizontal=zone.get("flip_horizontal", False),
            )
            for zone in data["zones"]
        )
        if not zones or len({zone.key for zone in zones}) != len(zones):
            raise ValueError("Tactile layout must contain unique zone keys")
        return cls(height=data["height"], width=data["width"], zones=zones)

    @classmethod
    def from_jq_skin(
        cls,
        parts: Sequence[str] = ("upper_body",),
        output_size: tuple[int, int] = (128, 128),
        margin: int = 2,
    ) -> "TactileLayout":
        """Build the same anatomical layout used by jq_skin's visualizer."""

        # Imported lazily so processor tests work without tactile hardware packages.
        from jq_skin.layout import BODY_PART_LAYOUTS, BodyPart
        from jq_skin.viz import ZONE_EXTENT

        canvas_h, canvas_w = output_size
        if canvas_h <= 2 * margin or canvas_w <= 2 * margin:
            raise ValueError("output_size must be larger than twice the margin")

        raw_zones: list[tuple[str, tuple[int, int], float, float, float, float, bool]] = []
        for part_name in parts:
            canonical_part = part_name.lower()
            body_part = BodyPart[canonical_part.upper()]
            layout = BODY_PART_LAYOUTS[body_part]
            for zone in layout.zones:
                x_first, x_last, y_bottom, y_top = ZONE_EXTENT[body_part][zone.name]
                left, right = sorted((float(x_first), float(x_last)))
                top, bottom = -float(y_top), -float(y_bottom)
                inset = 0.04 * min(right - left, bottom - top)
                raw_zones.append(
                    (
                        f"tactile_{canonical_part}_{zone.name}",
                        tuple(zone.grid_shape),
                        left + inset,
                        top + inset,
                        right - inset,
                        bottom - inset,
                        x_first > x_last,
                    )
                )

        if not raw_zones:
            raise ValueError("No tactile zones found for the requested body parts")

        min_x = min(zone[2] for zone in raw_zones)
        min_y = min(zone[3] for zone in raw_zones)
        max_x = max(zone[4] for zone in raw_zones)
        max_y = max(zone[5] for zone in raw_zones)
        source_w = max_x - min_x
        source_h = max_y - min_y
        scale = min((canvas_w - 2 * margin) / source_w, (canvas_h - 2 * margin) / source_h)
        x_offset = (canvas_w - source_w * scale) / 2.0
        y_offset = (canvas_h - source_h * scale) / 2.0

        placements = []
        for key, grid_shape, left, top, right, bottom, mirrored in raw_zones:
            x0 = int(round(x_offset + (left - min_x) * scale))
            y0 = int(round(y_offset + (top - min_y) * scale))
            x1 = int(round(x_offset + (right - min_x) * scale))
            y1 = int(round(y_offset + (bottom - min_y) * scale))
            x0 = max(0, min(x0, canvas_w - 1))
            y0 = max(0, min(y0, canvas_h - 1))
            x1 = max(x0 + 1, min(x1, canvas_w))
            y1 = max(y0 + 1, min(y1, canvas_h))
            placements.append(
                ZonePlacement(
                    key=key,
                    grid_shape=grid_shape,
                    top=y0,
                    left=x0,
                    height=y1 - y0,
                    width=x1 - x0,
                    flip_horizontal=mirrored,
                )
            )
        return cls(height=canvas_h, width=canvas_w, zones=tuple(placements))

    @classmethod
    def packed_from_readings(
        cls,
        readings: Mapping[str, np.ndarray],
        output_size: tuple[int, int] = (128, 128),
    ) -> "TactileLayout":
        """Build a deterministic fallback layout when jq_skin is unavailable."""

        zone_items = []
        for key, raw in readings.items():
            logical_key = key.removeprefix("state.")
            if not logical_key.startswith("tactile_"):
                continue
            vector = np.asarray(raw)
            if vector.ndim > 1:
                vector = vector[-1]
            channel_count = vector.size
            rows = max(1, math.isqrt(channel_count))
            while channel_count % rows:
                rows -= 1
            zone_items.append((logical_key, (rows, channel_count // rows)))
        zone_items.sort(key=lambda item: item[0])
        if not zone_items:
            raise ValueError("No tactile_* readings are available to construct a layout")

        canvas_h, canvas_w = output_size
        tile_columns = math.ceil(math.sqrt(len(zone_items)))
        tile_rows = math.ceil(len(zone_items) / tile_columns)
        placements = []
        for index, (key, grid_shape) in enumerate(zone_items):
            tile_row, tile_column = divmod(index, tile_columns)
            top = round(tile_row * canvas_h / tile_rows)
            bottom = round((tile_row + 1) * canvas_h / tile_rows)
            left = round(tile_column * canvas_w / tile_columns)
            right = round((tile_column + 1) * canvas_w / tile_columns)
            placements.append(
                ZonePlacement(
                    key=key,
                    grid_shape=grid_shape,
                    top=top,
                    left=left,
                    height=bottom - top,
                    width=right - left,
                )
            )
        return cls(canvas_h, canvas_w, tuple(placements))


def _resize_nearest(array: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Dependency-free nearest-neighbour resize for a two-dimensional array."""

    out_h, out_w = output_shape
    row_index = np.minimum(
        (np.arange(out_h, dtype=np.float64) * array.shape[0] / out_h).astype(np.int64),
        array.shape[0] - 1,
    )
    col_index = np.minimum(
        (np.arange(out_w, dtype=np.float64) * array.shape[1] / out_w).astype(np.int64),
        array.shape[1] - 1,
    )
    return array[row_index[:, None], col_index[None, :]]


class TactileImageProcessor:
    """Rasterize current-frame readings as ``(values, validity_mask)``.

    Both arrays have shape ``[1, H, W]`` and dtype ``float32``. ``readings``
    normally contains the logical state keys emitted by the SONIC dataset.
    """

    def __init__(
        self,
        layout: TactileLayout,
        value_min: float = 0.0,
        value_max: float = 255.0,
    ) -> None:
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        self.layout = layout
        self.value_min = float(value_min)
        self.value_max = float(value_max)

    @property
    def zone_keys(self) -> tuple[str, ...]:
        return tuple(zone.key for zone in self.layout.zones)

    def __call__(self, readings: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        values_canvas = np.zeros((self.layout.height, self.layout.width), dtype=np.float32)
        mask_canvas = np.zeros_like(values_canvas)
        scale = self.value_max - self.value_min

        for zone in self.layout.zones:
            raw = self._lookup(readings, zone.key)
            if raw is None:
                continue
            vector = np.asarray(raw, dtype=np.float32)
            if vector.ndim > 1:
                vector = vector[-1]
            vector = vector.reshape(-1)
            expected = int(np.prod(zone.grid_shape))
            if vector.size != expected:
                raise ValueError(
                    f"Tactile zone {zone.key!r} has {vector.size} values; expected {expected}"
                )

            finite = np.isfinite(vector)
            normalized = (np.where(finite, vector, self.value_min) - self.value_min) / scale
            grid = np.clip(normalized, 0.0, 1.0).reshape(zone.grid_shape)
            valid_grid = finite.astype(np.float32).reshape(zone.grid_shape)
            if zone.flip_horizontal:
                grid = grid[:, ::-1]
                valid_grid = valid_grid[:, ::-1]

            target = (zone.height, zone.width)
            resized = _resize_nearest(grid, target)
            resized_mask = _resize_nearest(valid_grid, target)
            rows = slice(zone.top, zone.top + zone.height)
            cols = slice(zone.left, zone.left + zone.width)
            values_canvas[rows, cols] = resized
            mask_canvas[rows, cols] = resized_mask

        return values_canvas[None], mask_canvas[None]

    @staticmethod
    def _lookup(readings: Mapping[str, np.ndarray], logical_key: str):
        if logical_key in readings:
            return readings[logical_key]
        state_key = f"state.{logical_key}"
        if state_key in readings:
            return readings[state_key]
        if logical_key.startswith("tactile_"):
            suffix = logical_key.removeprefix("tactile_")
            for key, value in readings.items():
                if key.startswith("observation.tactile."):
                    flattened = key.removeprefix("observation.tactile.").replace(".", "_", 1)
                    if flattened == suffix:
                        return value
        return None
