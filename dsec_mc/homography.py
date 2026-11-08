from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .affine import _event_image
from .algorithm import contrast_objective
from .blockwise import (
    BlockwiseTranslation,
    phase_correlation_shift,
    warp_event_volume,
)


@dataclass(frozen=True)
class HomographyField:
    homography_grid: np.ndarray
    correlation_grid: np.ndarray
    residual_grid: np.ndarray
    source_grid: np.ndarray
    center_y: np.ndarray
    center_x: np.ndarray
    dense_vx: np.ndarray
    dense_vy: np.ndarray

    def interpolate(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the dense endpoint-displacement fields used for warping."""
        if self.dense_vx.shape != (height, width) or self.dense_vy.shape != (height, width):
            raise ValueError("requested field shape does not match the stored result")
        return self.dense_vx, self.dense_vy


def _homography_displacement(
    homography: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    time_delta: float,
    min_denominator: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an early-to-late homography to endpoint displacement."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        shape = np.shape(x)
        invalid = np.zeros(shape, dtype=bool)
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            invalid,
        )

    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) >= min_denominator)
    safe_denominator = np.where(valid, denominator, 1.0)
    projected_x = (
        matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    ) / safe_denominator
    projected_y = (
        matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    ) / safe_denominator
    vx = (projected_x - x) / time_delta
    vy = (projected_y - y) / time_delta
    valid &= np.isfinite(vx) & np.isfinite(vy)
    return vx.astype(np.float32), vy.astype(np.float32), valid


def estimate_homography_from_event_groups(
    early: np.ndarray,
    late: np.ndarray,
    *,
    time_delta: float,
    initial_endpoint: tuple[float, float],
    min_denominator: float = 0.10,
) -> tuple[np.ndarray | None, float]:
    """Estimate a local early-to-late homography with ECC."""
    template = _event_image(early)
    moving = _event_image(late)
    mask = ((template > 0.02) | (moving > 0.02)).astype(np.uint8)
    if np.count_nonzero(mask) < max(24, template.size // 50):
        return None, 0.0

    homography = np.asarray(
        [
            [1.0, 0.0, initial_endpoint[0] * time_delta],
            [0.0, 1.0, initial_endpoint[1] * time_delta],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    try:
        correlation, homography = cv2.findTransformECC(
            template,
            moving,
            homography,
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            mask,
            5,
        )
    except cv2.error:
        return None, 0.0

    if not np.all(np.isfinite(homography)):
        return None, 0.0
    if abs(float(homography[2, 2])) < min_denominator:
        return None, 0.0
    homography = homography.astype(np.float64) / float(homography[2, 2])
    return homography, float(correlation)


def estimate_local_homography_field(
    volume: np.ndarray,
    blockwise: BlockwiseTranslation,
    *,
    base_vx_field: np.ndarray,
    base_vy_field: np.ndarray,
    height: int,
    width: int,
    region_size: int = 160,
    stride: int = 80,
    phase_bins: int = 3,
    min_correlation: float = 0.30,
    min_contrast_gain: float = 1.02,
    max_local_variation: float = 32.0,
    max_abs_shift: float = 48.0,
    min_denominator: float = 0.10,
) -> HomographyField:
    """
    Estimate overlapping local event homographies and fill weak regions.

    Each accepted homography is converted to an endpoint displacement field
    before blending. Rejected regions remain exactly on the block-wise field.
    """
    bins = volume.shape[0]
    phase_bins = min(max(1, phase_bins), bins // 2)
    early = volume[:phase_bins]
    late = volume[-phase_bins:]
    early_center = float(np.mean((np.arange(phase_bins) + 0.5) / bins))
    late_center = float(np.mean((np.arange(bins - phase_bins, bins) + 0.5) / bins))
    time_delta = late_center - early_center

    region_size = min(region_size, height, width)
    stride = min(stride, region_size)
    y_starts = list(range(0, height - region_size + 1, stride))
    if y_starts[-1] != height - region_size:
        y_starts.append(height - region_size)
    x_starts = list(range(0, width - region_size + 1, stride))
    if x_starts[-1] != width - region_size:
        x_starts.append(width - region_size)
    center_y = np.asarray([start + (region_size - 1) / 2 for start in y_starts])
    center_x = np.asarray([start + (region_size - 1) / 2 for start in x_starts])

    homography_grid = np.repeat(
        np.eye(3, dtype=np.float32)[None, None, :, :],
        len(y_starts) * len(x_starts),
        axis=0,
    ).reshape(len(y_starts), len(x_starts), 3, 3)
    correlation_grid = np.full((len(y_starts), len(x_starts)), np.nan, dtype=np.float32)
    residual_grid = np.full_like(correlation_grid, np.nan)
    source_grid = np.full(correlation_grid.shape, "fallback", dtype="<U20")
    dense_vx = base_vx_field.copy()
    dense_vy = base_vy_field.copy()
    temporal_factor = (bins - 1) / (2.0 * bins)
    local_y, local_x = np.indices((region_size, region_size), dtype=np.float32)
    corner_x = np.asarray(
        [0.0, region_size - 1.0, 0.0, region_size - 1.0, (region_size - 1.0) / 2.0]
    )
    corner_y = np.asarray(
        [0.0, 0.0, region_size - 1.0, region_size - 1.0, (region_size - 1.0) / 2.0]
    )

    for iy, y0 in enumerate(y_starts):
        for ix, x0 in enumerate(x_starts):
            x1 = x0 + region_size
            y1 = y0 + region_size
            block_y0 = int(np.searchsorted(blockwise.center_y, y0, side="left"))
            block_y1 = int(np.searchsorted(blockwise.center_y, y1, side="right"))
            block_x0 = int(np.searchsorted(blockwise.center_x, x0, side="left"))
            block_x1 = int(np.searchsorted(blockwise.center_x, x1, side="right"))
            local_valid = blockwise.valid_grid[block_y0:block_y1, block_x0:block_x1]
            if np.any(local_valid):
                local_vx = blockwise.vx_grid[block_y0:block_y1, block_x0:block_x1][
                    local_valid
                ]
                local_vy = blockwise.vy_grid[block_y0:block_y1, block_x0:block_x1][
                    local_valid
                ]
                initial = (float(np.median(local_vx)), float(np.median(local_vy)))
            else:
                initial = (blockwise.fallback_vx, blockwise.fallback_vy)

            early_region = early[:, y0:y1, x0:x1]
            late_region = late[:, y0:y1, x0:x1]
            phase_dx, phase_dy, phase_score = phase_correlation_shift(
                _event_image(early_region),
                _event_image(late_region),
            )
            phase_endpoint = (phase_dx / time_delta, phase_dy / time_delta)
            if (
                phase_score >= 2.0
                and np.linalg.norm(phase_endpoint) <= max_abs_shift
            ):
                initial = phase_endpoint

            homography, correlation = estimate_homography_from_event_groups(
                early_region,
                late_region,
                time_delta=time_delta,
                initial_endpoint=initial,
                min_denominator=min_denominator,
            )
            valid_model = homography is not None and correlation >= min_correlation
            if valid_model:
                homography_grid[iy, ix] = homography.astype(np.float32)
                corner_vx, corner_vy, corner_valid = _homography_displacement(
                    homography,
                    corner_x,
                    corner_y,
                    time_delta=time_delta,
                    min_denominator=min_denominator,
                )
                center_displacement = np.asarray(
                    [corner_vx[-1], corner_vy[-1]],
                    dtype=np.float64,
                )
                variation = np.linalg.norm(
                    np.stack([corner_vx[:-1], corner_vy[:-1]], axis=-1)
                    - center_displacement,
                    axis=1,
                )
                valid_model = (
                    np.all(corner_valid)
                    and np.all(np.isfinite(variation))
                    and np.linalg.norm(center_displacement) <= max_abs_shift
                    and np.max(variation) <= max_local_variation
                )

            contrast_gain = 0.0
            displacement_valid = False
            homography_vx = np.zeros((region_size, region_size), dtype=np.float32)
            homography_vy = np.zeros_like(homography_vx)
            if valid_model:
                homography_vx, homography_vy, displacement_valid = _homography_displacement(
                    homography,
                    local_x,
                    local_y,
                    time_delta=time_delta,
                    min_denominator=min_denominator,
                )
                displacement_valid &= (
                    np.abs(homography_vx) <= max_abs_shift
                ) & (np.abs(homography_vy) <= max_abs_shift)
                valid_model = bool(np.all(displacement_valid))

            if valid_model:
                block_vx_local = base_vx_field[y0:y1, x0:x1]
                block_vy_local = base_vy_field[y0:y1, x0:x1]
                max_local_shift = temporal_factor * max(
                    float(np.max(np.abs(homography_vx))),
                    float(np.max(np.abs(homography_vy))),
                    float(np.max(np.abs(block_vx_local))),
                    float(np.max(np.abs(block_vy_local))),
                )
                border = int(np.ceil(max_local_shift)) + 1
                if region_size - 2 * border >= 16:
                    region_volume = volume[:, y0:y1, x0:x1]
                    homography_image, _ = warp_event_volume(
                        region_volume,
                        homography_vx,
                        homography_vy,
                    )
                    block_image, _ = warp_event_volume(
                        region_volume,
                        block_vx_local,
                        block_vy_local,
                    )
                    homography_score = contrast_objective(
                        homography_image[border:-border, border:-border]
                    )
                    block_score = contrast_objective(
                        block_image[border:-border, border:-border]
                    )
                    contrast_gain = homography_score / max(block_score, 1e-8)
                    valid_model = contrast_gain >= min_contrast_gain
                else:
                    valid_model = False

            if valid_model:
                correlation_grid[iy, ix] = correlation
                residual_grid[iy, ix] = max(0.0, 1.0 - correlation)
                source_grid[iy, ix] = "ecc-homography"
                edge_distance = np.minimum(
                    np.minimum(local_y, region_size - 1 - local_y),
                    np.minimum(local_x, region_size - 1 - local_x),
                )
                taper = np.clip(edge_distance / (0.20 * region_size), 0.0, 1.0)
                strength = min(
                    1.0,
                    max(0.20, (contrast_gain - 1.0) / 0.08),
                ) * min(1.0, correlation)
                weight = taper * strength
                dense_vx[y0:y1, x0:x1] = (
                    dense_vx[y0:y1, x0:x1] * (1.0 - weight)
                    + homography_vx * weight
                )
                dense_vy[y0:y1, x0:x1] = (
                    dense_vy[y0:y1, x0:x1] * (1.0 - weight)
                    + homography_vy * weight
                )

    return HomographyField(
        homography_grid=homography_grid,
        correlation_grid=correlation_grid,
        residual_grid=residual_grid,
        source_grid=source_grid,
        center_y=center_y,
        center_x=center_x,
        dense_vx=dense_vx.astype(np.float32),
        dense_vy=dense_vy.astype(np.float32),
    )
