from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (
    distance_transform_edt,
    map_coordinates,
    median_filter,
)

from .algorithm import contrast_objective, refocus_event_slices


@dataclass(frozen=True)
class BlockwiseTranslation:
    vx_grid: np.ndarray
    vy_grid: np.ndarray
    confidence_grid: np.ndarray
    valid_grid: np.ndarray
    center_y: np.ndarray
    center_x: np.ndarray
    fallback_vx: float
    fallback_vy: float

    def interpolate(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Return dense endpoint-displacement fields."""
        if len(self.center_y) == 1:
            return (
                np.full((height, width), self.vx_grid[0, 0], dtype=np.float32),
                np.full((height, width), self.vy_grid[0, 0], dtype=np.float32),
            )

        x_points = np.concatenate(([0.0], self.center_x, [width - 1.0]))
        y_points = np.concatenate(([0.0], self.center_y, [height - 1.0]))
        vx_values = np.pad(
            self.vx_grid,
            ((1, 1), (1, 1)),
            mode="edge",
        )
        vy_values = np.pad(
            self.vy_grid,
            ((1, 1), (1, 1)),
            mode="edge",
        )
        vx_spline = RectBivariateSpline(y_points, x_points, vx_values, kx=1, ky=1)
        vy_spline = RectBivariateSpline(y_points, x_points, vy_values, kx=1, ky=1)
        yy, xx = np.indices((height, width), dtype=np.float32)
        vx = vx_spline.ev(yy.ravel(), xx.ravel()).reshape(height, width)
        vy = vy_spline.ev(yy.ravel(), xx.ravel()).reshape(height, width)
        return vx.astype(np.float32), vy.astype(np.float32)


def slice_event_images(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    normalized_time: np.ndarray,
    *,
    width: int,
    height: int,
    bins: int = 9,
) -> np.ndarray:
    """Create a signed polarity-count volume with shape [bins, height, width]."""
    x_index = np.rint(x).astype(np.int32)
    y_index = np.rint(y).astype(np.int32)
    keep = (
        (x_index >= 0)
        & (x_index < width)
        & (y_index >= 0)
        & (y_index < height)
    )
    x_index = x_index[keep]
    y_index = y_index[keep]
    time_index = np.minimum(
        (normalized_time[keep] * bins).astype(np.int32),
        bins - 1,
    )
    signed = np.where(polarity[keep] > 0, 1.0, -1.0).astype(np.float32)
    volume = np.zeros((bins, height, width), dtype=np.float32)
    np.add.at(volume, (time_index, y_index, x_index), signed)
    return volume


def _remove_mean_window(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    centered = image - float(np.mean(image))
    return centered * window


def _parabolic_peak_offset(
    left: float,
    center: float,
    right: float,
) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))


def phase_correlation_shift(
    reference: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimate the displacement from reference to moving.

    A positive dx means image content moved right. The returned confidence is a
    peak-to-background score and is diagnostic rather than a probability.
    """
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("phase correlation expects two equal-size 2-D images")

    ref = _remove_mean_window(reference)
    mov = _remove_mean_window(moving)
    spectrum = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(spectrum)
    spectrum /= np.maximum(magnitude, 1e-8)
    correlation = np.fft.ifft2(spectrum).real

    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak = float(correlation[peak_y, peak_x])
    background = float(np.median(correlation))
    spread = float(np.std(correlation))
    confidence = (peak - background) / max(spread, 1e-8)

    dy = float(peak_y if peak_y <= reference.shape[0] // 2 else peak_y - reference.shape[0])
    dx = float(peak_x if peak_x <= reference.shape[1] // 2 else peak_x - reference.shape[1])

    if 0 < peak_y < correlation.shape[0] - 1:
        dy += _parabolic_peak_offset(
            float(correlation[peak_y - 1, peak_x]),
            float(correlation[peak_y, peak_x]),
            float(correlation[peak_y + 1, peak_x]),
        )
    if 0 < peak_x < correlation.shape[1] - 1:
        dx += _parabolic_peak_offset(
            float(correlation[peak_y, peak_x - 1]),
            float(correlation[peak_y, peak_x]),
            float(correlation[peak_y, peak_x + 1]),
        )
    # The inverse FFT peak is the shift that aligns `moving` back to
    # `reference`; negate it to describe content motion.
    return -dx, -dy, confidence


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size > length:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _refine_tile_translation(
    slices: list[np.ndarray],
    initial: tuple[float, float],
    *,
    refine_radius: int,
    max_abs_shift: float,
) -> tuple[float, float, float, float]:
    local_bound = int(np.ceil(max(abs(initial[0]), abs(initial[1])))) + refine_radius
    if local_bound >= min(slices[0].shape) / 2 - 2:
        return initial[0], initial[1], 1.0, 0.0

    candidates_x = np.arange(
        max(-max_abs_shift, initial[0] - refine_radius),
        min(max_abs_shift, initial[0] + refine_radius) + 0.5,
        1.0,
    )
    candidates_y = np.arange(
        max(-max_abs_shift, initial[1] - refine_radius),
        min(max_abs_shift, initial[1] + refine_radius) + 0.5,
        1.0,
    )
    if candidates_x.size == 0 or candidates_y.size == 0:
        return initial[0], initial[1], 1.0, 0.0

    scores = np.empty((candidates_y.size, candidates_x.size), dtype=np.float32)
    for iy, vy in enumerate(candidates_y):
        for ix, vx in enumerate(candidates_x):
            image = refocus_event_slices(
                slices,
                (float(vx), float(vy)),
                max_shift=local_bound,
            )
            scores[iy, ix] = contrast_objective(image)

    best = np.unravel_index(int(np.argmax(scores)), scores.shape)
    vx = float(candidates_x[best[1]])
    vy = float(candidates_y[best[0]])
    baseline_image = refocus_event_slices(
        slices,
        (0.0, 0.0),
        max_shift=local_bound,
    )
    baseline_score = contrast_objective(baseline_image)
    gain = float(scores[best] / max(baseline_score, 1e-8))
    return vx, vy, gain, local_bound


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if np.all(valid):
        return values.copy()
    if not np.any(valid):
        raise ValueError("No valid blocks were available for interpolation.")
    _, indices = distance_transform_edt(~valid, return_indices=True)
    return values[tuple(indices)]


def estimate_blockwise_translation(
    volume: np.ndarray,
    *,
    tile_size: int = 128,
    stride: int = 64,
    phase_bins: int = 3,
    refine_radius: int = 3,
    max_abs_shift: float = 48.0,
    min_events: int = 160,
    fallback: tuple[float, float] | None = None,
) -> BlockwiseTranslation:
    """Estimate an overlapping tile grid and return a piecewise-smooth field."""
    bins, height, width = volume.shape
    if bins < 4:
        raise ValueError("At least four temporal bins are required.")
    phase_bins = min(max(1, phase_bins), bins // 2)
    early = np.abs(volume[:phase_bins]).sum(axis=0)
    late = np.abs(volume[-phase_bins:]).sum(axis=0)
    early_center = np.mean((np.arange(phase_bins) + 0.5) / bins)
    late_center = np.mean((np.arange(bins - phase_bins, bins) + 0.5) / bins)
    time_delta = late_center - early_center

    tile_size = min(tile_size, height, width)
    y_starts = _tile_starts(height, tile_size, min(stride, tile_size))
    x_starts = _tile_starts(width, tile_size, min(stride, tile_size))
    center_y = np.asarray([start + (tile_size - 1) / 2 for start in y_starts])
    center_x = np.asarray([start + (tile_size - 1) / 2 for start in x_starts])
    vx_grid = np.zeros((len(y_starts), len(x_starts)), dtype=np.float32)
    vy_grid = np.zeros_like(vx_grid)
    confidence = np.zeros_like(vx_grid)
    valid = np.zeros_like(vx_grid, dtype=bool)
    phase_vectors: list[tuple[float, float]] = []

    for iy, y0 in enumerate(y_starts):
        for ix, x0 in enumerate(x_starts):
            tile = volume[:, y0:y0 + tile_size, x0:x0 + tile_size]
            signed_count = int(np.count_nonzero(tile))
            early_tile = early[y0:y0 + tile_size, x0:x0 + tile_size]
            late_tile = late[y0:y0 + tile_size, x0:x0 + tile_size]
            phase_events = int(np.count_nonzero(early_tile) + np.count_nonzero(late_tile))
            phase_dx, phase_dy, phase_score = phase_correlation_shift(
                early_tile,
                late_tile,
            )
            phase_vx = phase_dx / max(time_delta, 1e-8)
            phase_vy = phase_dy / max(time_delta, 1e-8)
            phase_vectors.append((phase_vx, phase_vy))

            if signed_count < min_events:
                continue
            if phase_events < min_events // 2 or phase_score < 2.0:
                if fallback is None:
                    continue
                initial = fallback
            else:
                initial = (phase_vx, phase_vy)

            tile_slices = [item for item in tile]
            refined_vx, refined_vy, contrast_gain, _bound = _refine_tile_translation(
                tile_slices,
                initial,
                refine_radius=refine_radius,
                max_abs_shift=max_abs_shift,
            )
            vx_grid[iy, ix] = refined_vx
            vy_grid[iy, ix] = refined_vy
            event_confidence = min(1.0, signed_count / max(float(tile_size * tile_size), 1.0))
            phase_confidence = min(1.0, max(0.0, (phase_score - 2.0) / 8.0))
            contrast_confidence = min(1.0, max(0.0, (contrast_gain - 1.0) / 2.0))
            confidence[iy, ix] = (
                0.25 * event_confidence
                + 0.35 * phase_confidence
                + 0.40 * contrast_confidence
            )
            valid[iy, ix] = True

    if not np.any(valid):
        if fallback is None:
            raise ValueError("No block had enough events to estimate translation.")
        return BlockwiseTranslation(
            vx_grid=np.full_like(vx_grid, fallback[0]),
            vy_grid=np.full_like(vy_grid, fallback[1]),
            confidence_grid=np.ones_like(vx_grid),
            valid_grid=np.ones_like(valid),
            center_y=center_y,
            center_x=center_x,
            fallback_vx=float(fallback[0]),
            fallback_vy=float(fallback[1]),
        )

    # Use phase vectors as a global prior when the caller did not provide one.
    if fallback is None:
        phase_array = np.asarray(phase_vectors, dtype=np.float32)
        fallback = (
            float(np.median(phase_array[:, 0])),
            float(np.median(phase_array[:, 1])),
        )

    vx_filled = _nearest_fill(vx_grid, valid)
    vy_filled = _nearest_fill(vy_grid, valid)
    vx_smoothed = median_filter(vx_filled, size=3, mode="nearest")
    vy_smoothed = median_filter(vy_filled, size=3, mode="nearest")

    # Replace low-confidence outliers, but preserve confident local motion.
    deviation = np.hypot(vx_filled - vx_smoothed, vy_filled - vy_smoothed)
    replace = valid & (confidence < 0.45) & (deviation > 4.0)
    vx_grid[replace] = vx_smoothed[replace]
    vy_grid[replace] = vy_smoothed[replace]

    vx_filled = _nearest_fill(vx_grid, valid)
    vy_filled = _nearest_fill(vy_grid, valid)
    return BlockwiseTranslation(
        vx_grid=vx_filled,
        vy_grid=vy_filled,
        confidence_grid=confidence,
        valid_grid=valid,
        center_y=center_y,
        center_x=center_x,
        fallback_vx=float(fallback[0]),
        fallback_vy=float(fallback[1]),
    )


def warp_event_volume(
    volume: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp every temporal slice to the window midpoint."""
    bins, height, width = volume.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    compensated = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for index, image in enumerate(volume):
        tau_mid = (index + 0.5) / bins - 0.5
        shift_x = -vx_field * tau_mid
        shift_y = -vy_field * tau_mid
        sample_y = yy - shift_y
        sample_x = xx - shift_x
        warped = map_coordinates(
            image,
            [sample_y, sample_x],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        inside = (
            (sample_x >= 0)
            & (sample_x <= width - 1)
            & (sample_y >= 0)
            & (sample_y <= height - 1)
        )
        compensated += warped
        coverage += inside
    return compensated, coverage / bins


def warp_events_with_field(
    x: np.ndarray,
    y: np.ndarray,
    normalized_time: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp raw events with a dense endpoint-displacement field."""
    coordinates = [y, x]
    local_vx = map_coordinates(
        vx_field,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    local_vy = map_coordinates(
        vy_field,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    coefficient = -(normalized_time - 0.5)
    x_ref = x + local_vx * coefficient
    y_ref = y + local_vy * coefficient
    height, width = vx_field.shape
    inside = (
        (x_ref >= 0)
        & (x_ref < width)
        & (y_ref >= 0)
        & (y_ref < height)
    )
    return x_ref, y_ref, inside
