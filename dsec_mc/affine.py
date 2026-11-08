from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from .algorithm import contrast_objective
from .blockwise import (
    BlockwiseTranslation,
    phase_correlation_shift,
    warp_event_volume,
)


@dataclass(frozen=True)
class AffineField:
    coefficient_x: np.ndarray
    coefficient_y: np.ndarray
    residual_grid: np.ndarray
    correlation_grid: np.ndarray
    source_grid: np.ndarray
    center_y: np.ndarray
    center_x: np.ndarray
    dense_vx: np.ndarray | None = None
    dense_vy: np.ndarray | None = None

    def interpolate(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate local affine coefficients and evaluate the displacement."""
        if self.dense_vx is not None and self.dense_vy is not None:
            return self.dense_vx, self.dense_vy
        if len(self.center_y) == 1:
            coefficients = np.stack(
                [self.coefficient_x[0, 0], self.coefficient_y[0, 0]],
                axis=-1,
            )
            yy, xx = np.indices((height, width), dtype=np.float32)
            u = 2.0 * xx / max(width - 1, 1) - 1.0
            v = 2.0 * yy / max(height - 1, 1) - 1.0
            design = np.stack([u, v, np.ones_like(u)], axis=-1)
            displacement = design @ coefficients
            return displacement[:, :, 0], displacement[:, :, 1]

        x_points = np.concatenate(([0.0], self.center_x, [width - 1.0]))
        y_points = np.concatenate(([0.0], self.center_y, [height - 1.0]))
        yy, xx = np.indices((height, width), dtype=np.float32)
        u = 2.0 * xx / max(width - 1, 1) - 1.0
        v = 2.0 * yy / max(height - 1, 1) - 1.0
        design_fields = (u, v, np.ones_like(u))
        vx = np.zeros((height, width), dtype=np.float32)
        vy = np.zeros((height, width), dtype=np.float32)

        for coefficient_index, design in enumerate(design_fields):
            x_values = np.pad(
                self.coefficient_x[:, :, coefficient_index],
                ((1, 1), (1, 1)),
                mode="edge",
            )
            y_values = np.pad(
                self.coefficient_y[:, :, coefficient_index],
                ((1, 1), (1, 1)),
                mode="edge",
            )
            x_spline = RectBivariateSpline(
                y_points,
                x_points,
                x_values,
                kx=1,
                ky=1,
            )
            y_spline = RectBivariateSpline(
                y_points,
                x_points,
                y_values,
                kx=1,
                ky=1,
            )
            x_coefficient = x_spline.ev(yy.ravel(), xx.ravel()).reshape(height, width)
            y_coefficient = y_spline.ev(yy.ravel(), xx.ravel()).reshape(height, width)
            vx += x_coefficient * design
            vy += y_coefficient * design
        return vx.astype(np.float32), vy.astype(np.float32)


def _normalised_design(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    u = 2.0 * x / max(width - 1, 1) - 1.0
    v = 2.0 * y / max(height - 1, 1) - 1.0
    return np.stack([u, v, np.ones_like(u)], axis=-1)


def _robust_affine_fit(
    design: np.ndarray,
    displacement: np.ndarray,
    base_weight: np.ndarray,
    *,
    huber_delta: float,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    weights = base_weight.copy()
    coefficients = np.zeros((3, 2), dtype=np.float64)
    for _ in range(iterations):
        square_root = np.sqrt(np.maximum(weights, 1e-8))
        weighted_design = design * square_root[:, None]
        weighted_displacement = displacement * square_root[:, None]
        coefficients, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_displacement,
            rcond=None,
        )
        residuals = np.linalg.norm(design @ coefficients - displacement, axis=1)
        robust_weight = np.minimum(1.0, huber_delta / np.maximum(residuals, 1e-8))
        weights = base_weight * robust_weight
    final_residuals = np.linalg.norm(design @ coefficients - displacement, axis=1)
    return coefficients, final_residuals


def fit_local_affine_from_blocks(
    blockwise: BlockwiseTranslation,
    *,
    height: int,
    width: int,
    neighbourhood_radius: int = 3,
    huber_delta: float = 4.0,
    max_residual: float = 8.0,
    max_local_variation: float = 24.0,
    max_abs_shift: float = 48.0,
) -> AffineField:
    """
    Fit overlapping robust affine models to reliable block displacements.

    Each local model uses normalised image coordinates. If the sample
    distribution is degenerate or the fitted model is poorly supported, that
    centre falls back to the original block translation.
    """
    y_count, x_count = blockwise.vx_grid.shape
    coefficient_x = np.zeros((y_count, x_count, 3), dtype=np.float32)
    coefficient_y = np.zeros_like(coefficient_x)
    residual_grid = np.full((y_count, x_count), np.nan, dtype=np.float32)
    source_grid = np.full((y_count, x_count), "fallback", dtype="<U16")
    stride_y = np.median(np.diff(blockwise.center_y)) if y_count > 1 else 1.0
    stride_x = np.median(np.diff(blockwise.center_x)) if x_count > 1 else 1.0

    for iy, cy in enumerate(blockwise.center_y):
        for ix, cx in enumerate(blockwise.center_x):
            y0 = max(0, iy - neighbourhood_radius)
            y1 = min(y_count, iy + neighbourhood_radius + 1)
            x0 = max(0, ix - neighbourhood_radius)
            x1 = min(x_count, ix + neighbourhood_radius + 1)
            local_valid = blockwise.valid_grid[y0:y1, x0:x1]
            if np.count_nonzero(local_valid) < 4:
                coefficient_x[iy, ix, 2] = blockwise.vx_grid[iy, ix]
                coefficient_y[iy, ix, 2] = blockwise.vy_grid[iy, ix]
                continue

            grid_y, grid_x = np.meshgrid(
                blockwise.center_y[y0:y1],
                blockwise.center_x[x0:x1],
                indexing="ij",
            )
            selected = local_valid.ravel()
            sample_x = grid_x.ravel()[selected]
            sample_y = grid_y.ravel()[selected]
            sample_vx = blockwise.vx_grid[y0:y1, x0:x1].ravel()[selected]
            sample_vy = blockwise.vy_grid[y0:y1, x0:x1].ravel()[selected]
            confidence = blockwise.confidence_grid[y0:y1, x0:x1].ravel()[selected]
            displacement = np.stack([sample_vx, sample_vy], axis=-1).astype(np.float64)
            design = _normalised_design(sample_x, sample_y, width, height)
            if np.linalg.matrix_rank(design, tol=1e-6) < 3:
                coefficient_x[iy, ix, 2] = blockwise.vx_grid[iy, ix]
                coefficient_y[iy, ix, 2] = blockwise.vy_grid[iy, ix]
                continue

            distance_y = (sample_y - cy) / max(stride_y, 1.0)
            distance_x = (sample_x - cx) / max(stride_x, 1.0)
            distance = np.hypot(distance_x, distance_y)
            base_weight = (
                0.25 + np.clip(confidence, 0.0, 1.0)
            ) * np.exp(-0.5 * (distance / max(neighbourhood_radius, 1.0)) ** 2)
            try:
                coefficients, residuals = _robust_affine_fit(
                    design,
                    displacement,
                    base_weight,
                    huber_delta=huber_delta,
                )
            except np.linalg.LinAlgError:
                coefficients = np.zeros((3, 2), dtype=np.float64)
                residuals = np.full(len(design), np.inf)

            center_design = _normalised_design(
                np.asarray([cx]),
                np.asarray([cy]),
                width,
                height,
            )[0]
            center_correction = center_design @ coefficients
            fallback_translation = np.asarray(
                [blockwise.vx_grid[iy, ix], blockwise.vy_grid[iy, ix]],
                dtype=np.float64,
            )
            local_extent = neighbourhood_radius * max(stride_x, stride_y)
            extent_design = _normalised_design(
                np.asarray([cx - local_extent, cx + local_extent, cx, cx]),
                np.asarray([cy, cy, cy - local_extent, cy + local_extent]),
                width,
                height,
            )
            extent_displacement = extent_design @ coefficients
            variation = np.linalg.norm(
                extent_displacement - center_correction,
                axis=1,
            )
            valid_model = (
                np.median(residuals) <= max_residual
                and np.max(variation) <= max_local_variation
                and np.linalg.norm(center_correction) <= max_abs_shift
            )
            if valid_model:
                coefficient_x[iy, ix] = coefficients[:, 0]
                coefficient_y[iy, ix] = coefficients[:, 1]
                residual_grid[iy, ix] = float(np.median(residuals))
                source_grid[iy, ix] = "affine"
            else:
                coefficient_x[iy, ix, 2] = fallback_translation[0]
                coefficient_y[iy, ix, 2] = fallback_translation[1]

    return AffineField(
        coefficient_x=coefficient_x,
        coefficient_y=coefficient_y,
        residual_grid=residual_grid,
        correlation_grid=np.where(
            source_grid == "affine",
            np.exp(-np.nan_to_num(residual_grid, nan=10.0)),
            np.nan,
        ).astype(np.float32),
        source_grid=source_grid,
        center_y=blockwise.center_y,
        center_x=blockwise.center_x,
        dense_vx=None,
        dense_vy=None,
    )


def _event_image(group: np.ndarray, *, sigma: float = 1.4) -> np.ndarray:
    image = np.abs(group).sum(axis=0).astype(np.float32)
    image = gaussian_filter(image, sigma=sigma)
    scale = float(np.percentile(image, 99.5))
    if scale <= 1e-8:
        return image
    return np.clip(image / scale, 0.0, 1.0).astype(np.float32)


def estimate_affine_from_event_groups(
    early: np.ndarray,
    late: np.ndarray,
    *,
    time_delta: float,
    initial_endpoint: tuple[float, float],
) -> tuple[np.ndarray | None, float]:
    """
    Estimate the endpoint affine displacement with ECC.

    OpenCV returns a warp from the template (early) to the input (late).
    Therefore the linear part is converted to an endpoint displacement field
    using `(A - I) / dt` and `b / dt`.
    """
    template = _event_image(early)
    moving = _event_image(late)
    mask = ((template > 0.02) | (moving > 0.02)).astype(np.uint8)
    if np.count_nonzero(mask) < max(24, template.size // 50):
        return None, 0.0

    warp = np.asarray(
        [
            [1.0, 0.0, initial_endpoint[0] * time_delta],
            [0.0, 1.0, initial_endpoint[1] * time_delta],
        ],
        dtype=np.float32,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
            mask,
            5,
        )
    except cv2.error:
        return None, 0.0
    linear = warp[:, :2].astype(np.float64)
    translation = warp[:, 2].astype(np.float64)
    linear_velocity = (linear - np.eye(2, dtype=np.float64)) / time_delta
    endpoint_coefficients = np.stack(
        [
            np.asarray([linear_velocity[0, 0], linear_velocity[1, 0]]),
            np.asarray([linear_velocity[0, 1], linear_velocity[1, 1]]),
            translation / time_delta,
        ],
        axis=0,
    )
    return endpoint_coefficients, float(correlation)


def estimate_local_affine_field(
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
    max_local_variation: float = 24.0,
    max_abs_shift: float = 48.0,
) -> AffineField:
    """Estimate overlapping local affine event motion and fill weak regions."""
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
    coefficient_x = np.zeros((len(y_starts), len(x_starts), 3), dtype=np.float32)
    coefficient_y = np.zeros_like(coefficient_x)
    residual_grid = np.full(coefficient_x.shape[:2], np.nan, dtype=np.float32)
    correlation_grid = np.full(coefficient_x.shape[:2], np.nan, dtype=np.float32)
    source_grid = np.full(coefficient_x.shape[:2], "fallback", dtype="<U16")
    dense_vx = base_vx_field.copy()
    dense_vy = base_vy_field.copy()
    temporal_factor = (bins - 1) / (2.0 * bins)

    for iy, y0 in enumerate(y_starts):
        for ix, x0 in enumerate(x_starts):
            cy = center_y[iy]
            cx = center_x[ix]
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
            phase_vx, phase_vy, phase_score = phase_correlation_shift(
                _event_image(early_region),
                _event_image(late_region),
            )
            phase_endpoint = (phase_vx / time_delta, phase_vy / time_delta)
            if (
                phase_score >= 2.0
                and np.linalg.norm(phase_endpoint) <= max_abs_shift
            ):
                initial = phase_endpoint

            endpoint_coefficients, correlation = estimate_affine_from_event_groups(
                early_region,
                late_region,
                time_delta=time_delta,
                initial_endpoint=initial,
            )
            if endpoint_coefficients is not None:
                local_extents = np.asarray(
                    [
                        [-region_size / 2, -region_size / 2, 1.0],
                        [region_size / 2, region_size / 2, 1.0],
                        [-region_size / 2, region_size / 2, 1.0],
                        [region_size / 2, -region_size / 2, 1.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                local_displacements = local_extents @ endpoint_coefficients
                center_displacement = local_displacements[-1]
                variation = np.linalg.norm(
                    local_displacements[:-1] - center_displacement,
                    axis=1,
                )
                valid_model = (
                    correlation >= min_correlation
                    and np.linalg.norm(center_displacement) <= max_abs_shift
                    and np.max(variation) <= max_local_variation
                )
            else:
                valid_model = False

            contrast_gain = 0.0
            if valid_model:
                local_y, local_x = np.indices(
                    (region_size, region_size),
                    dtype=np.float32,
                )
                affine_vx_local = (
                    endpoint_coefficients[0, 0] * local_x
                    + endpoint_coefficients[1, 0] * local_y
                    + endpoint_coefficients[2, 0]
                )
                affine_vy_local = (
                    endpoint_coefficients[0, 1] * local_x
                    + endpoint_coefficients[1, 1] * local_y
                    + endpoint_coefficients[2, 1]
                )
                block_vx_local = base_vx_field[y0:y1, x0:x1]
                block_vy_local = base_vy_field[y0:y1, x0:x1]
                max_local_shift = temporal_factor * max(
                    float(np.max(np.abs(affine_vx_local))),
                    float(np.max(np.abs(affine_vy_local))),
                    float(np.max(np.abs(block_vx_local))),
                    float(np.max(np.abs(block_vy_local))),
                )
                border = int(np.ceil(max_local_shift)) + 1
                if region_size - 2 * border >= 16:
                    region_volume = volume[:, y0:y1, x0:x1]
                    affine_image, _ = warp_event_volume(
                        region_volume,
                        affine_vx_local,
                        affine_vy_local,
                    )
                    block_image, _ = warp_event_volume(
                        region_volume,
                        block_vx_local,
                        block_vy_local,
                    )
                    affine_score = contrast_objective(
                        affine_image[border:-border, border:-border]
                    )
                    block_score = contrast_objective(
                        block_image[border:-border, border:-border]
                    )
                    contrast_gain = affine_score / max(block_score, 1e-8)
                    valid_model = contrast_gain >= min_contrast_gain
                else:
                    valid_model = False

            if valid_model:
                scale_x = (width - 1) / 2.0
                scale_y = (height - 1) / 2.0
                offset_x = scale_x - x0
                offset_y = scale_y - y0
                for output_index, target in (
                    (0, coefficient_x),
                    (1, coefficient_y),
                ):
                    local_linear_x = endpoint_coefficients[0, output_index]
                    local_linear_y = endpoint_coefficients[1, output_index]
                    local_offset = endpoint_coefficients[2, output_index]
                    target[iy, ix, 0] = local_linear_x * scale_x
                    target[iy, ix, 1] = local_linear_y * scale_y
                    target[iy, ix, 2] = (
                        local_linear_x * offset_x
                        + local_linear_y * offset_y
                        + local_offset
                    )
                correlation_grid[iy, ix] = correlation
                residual_grid[iy, ix] = max(0.0, 1.0 - correlation)
                source_grid[iy, ix] = "ecc-affine"
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
                    + affine_vx_local * weight
                )
                dense_vy[y0:y1, x0:x1] = (
                    dense_vy[y0:y1, x0:x1] * (1.0 - weight)
                    + affine_vy_local * weight
                )
            else:
                coefficient_x[iy, ix, 2] = initial[0]
                coefficient_y[iy, ix, 2] = initial[1]

    return AffineField(
        coefficient_x=coefficient_x,
        coefficient_y=coefficient_y,
        residual_grid=residual_grid,
        correlation_grid=correlation_grid,
        source_grid=source_grid,
        center_y=center_y,
        center_x=center_x,
        dense_vx=dense_vx.astype(np.float32),
        dense_vy=dense_vy.astype(np.float32),
    )
