from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift as image_shift


@dataclass(frozen=True)
class MotionEstimate:
    vx: float
    vy: float
    baseline_score: float
    compensated_score: float
    coarse_grid: np.ndarray
    search_vx: np.ndarray
    search_vy: np.ndarray


def slice_local_events(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    normalized_time: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    radius: int,
    bins: int = 9,
) -> tuple[list[np.ndarray], tuple[slice, slice]]:
    """Create signed polarity slices in a local patch."""
    if bins < 2:
        raise ValueError("At least two temporal bins are required.")
    size = 2 * radius + 1
    y0 = int(center_y - radius)
    x0 = int(center_x - radius)
    local_x_all = np.rint(x - x0).astype(np.int32)
    local_y_all = np.rint(y - y0).astype(np.int32)
    keep = (
        (local_x_all >= 0)
        & (local_x_all < size)
        & (local_y_all >= 0)
        & (local_y_all < size)
    )
    local_x = local_x_all[keep]
    local_y = local_y_all[keep]
    local_t = normalized_time[keep]
    signed = np.where(polarity[keep] > 0, 1.0, -1.0).astype(np.float32)
    bin_index = np.minimum((local_t * bins).astype(np.int32), bins - 1)

    slices: list[np.ndarray] = []
    for index in range(bins):
        selected = bin_index == index
        image = np.zeros((size, size), dtype=np.float32)
        np.add.at(image, (local_y[selected], local_x[selected]), signed[selected])
        slices.append(image)
    return slices, (slice(y0, y0 + size), slice(x0, x0 + size))


def _stable_crop_size(radius: int, max_shift: int) -> int:
    size = 2 * radius + 1
    if max_shift <= 0:
        return size
    return size - 2 * (int(np.ceil(max_shift)) + 1)


def contrast_objective(image: np.ndarray, sigma: float = 1.0) -> float:
    """Variance of a polarity-weighted, lightly smoothed event image."""
    smooth = gaussian_filter(image, sigma=sigma, mode="constant", cval=0.0)
    return float(np.var(smooth))


def refocus_event_slices(
    slices: list[np.ndarray],
    motion: tuple[float, float],
    *,
    max_shift: int | None = None,
    sigma: float = 1.0,
) -> np.ndarray:
    """Warp each time slice to the midpoint and accumulate it."""
    bins = len(slices)
    vx, vy = motion
    border = int(np.ceil(max_shift)) + 1 if max_shift is not None else 0
    first = slices[0]
    if border:
        target_shape = first[border:-border, border:-border].shape
    else:
        target_shape = first.shape
    accumulated = np.zeros(target_shape, dtype=np.float32)
    for index, image in enumerate(slices):
        tau_mid = (index + 0.5) / bins - 0.5
        shift_x = -vx * tau_mid
        shift_y = -vy * tau_mid
        warped = image_shift(
            image,
            shift=(shift_y, shift_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        if border:
            warped = warped[border:-border, border:-border]
        else:
            warped = warped
        accumulated += warped
    return accumulated


def _score_candidate(
    slices: list[np.ndarray],
    motion: tuple[float, float],
    max_shift: int,
    sigma: float,
) -> float:
    image = refocus_event_slices(
        slices,
        motion,
        max_shift=max_shift,
        sigma=sigma,
    )
    return contrast_objective(image, sigma=sigma)


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -0.75, 0.75))


def estimate_translation(
    slices: list[np.ndarray],
    *,
    radius: int,
    search_radius: int = 24,
    coarse_step: int = 4,
    fine_radius: int = 4,
    sigma: float = 1.0,
) -> MotionEstimate:
    """
    Estimate a constant endpoint displacement by maximizing event image variance.

    The returned (vx, vy) is the cumulative displacement over the full event
    window. Each slice is warped to the temporal midpoint as in the proposal.
    """
    if coarse_step < 1:
        raise ValueError("coarse_step must be positive.")
    max_shift = max(search_radius, fine_radius)
    crop_size = _stable_crop_size(radius, max_shift)
    if crop_size < 16:
        raise ValueError("Patch is too small for the requested search radius.")

    coarse_values = np.arange(
        -search_radius,
        search_radius + coarse_step,
        coarse_step,
        dtype=np.float32,
    )
    coarse_grid = np.empty((coarse_values.size, coarse_values.size), dtype=np.float32)
    for iy, vy in enumerate(coarse_values):
        for ix, vx in enumerate(coarse_values):
            coarse_grid[iy, ix] = _score_candidate(
                slices,
                (float(vx), float(vy)),
                max_shift,
                sigma,
            )

    best_flat = int(np.argmax(coarse_grid))
    best_iy, best_ix = np.unravel_index(best_flat, coarse_grid.shape)
    coarse_vx = float(coarse_values[best_ix])
    coarse_vy = float(coarse_values[best_iy])

    fine_values = np.arange(
        max(-search_radius, coarse_vx - fine_radius),
        min(search_radius, coarse_vx + fine_radius) + 0.5,
        1,
        dtype=np.float32,
    )
    fine_values_y = np.arange(
        max(-search_radius, coarse_vy - fine_radius),
        min(search_radius, coarse_vy + fine_radius) + 0.5,
        1,
        dtype=np.float32,
    )
    fine_grid = np.empty((fine_values_y.size, fine_values.size), dtype=np.float32)
    for iy, vy in enumerate(fine_values_y):
        for ix, vx in enumerate(fine_values):
            fine_grid[iy, ix] = _score_candidate(
                slices,
                (float(vx), float(vy)),
                max_shift,
                sigma,
            )

    best_flat = int(np.argmax(fine_grid))
    best_iy, best_ix = np.unravel_index(best_flat, fine_grid.shape)
    vx = float(fine_values[best_ix])
    vy = float(fine_values_y[best_iy])

    # Sub-pixel refinement from the integer local maximum.
    if 0 < best_ix < fine_values.size - 1:
        vx += _parabolic_offset(
            float(fine_grid[best_iy, best_ix - 1]),
            float(fine_grid[best_iy, best_ix]),
            float(fine_grid[best_iy, best_ix + 1]),
        )
    if 0 < best_iy < fine_values_y.size - 1:
        vy += _parabolic_offset(
            float(fine_grid[best_iy - 1, best_ix]),
            float(fine_grid[best_iy, best_ix]),
            float(fine_grid[best_iy + 1, best_ix]),
        )

    baseline = refocus_event_slices(slices, (0.0, 0.0), max_shift=max_shift, sigma=sigma)
    compensated = refocus_event_slices(slices, (vx, vy), max_shift=max_shift, sigma=sigma)
    return MotionEstimate(
        vx=float(vx),
        vy=float(vy),
        baseline_score=contrast_objective(baseline, sigma=sigma),
        compensated_score=contrast_objective(compensated, sigma=sigma),
        coarse_grid=coarse_grid,
        search_vx=coarse_values,
        search_vy=coarse_values,
    )


def sharpness_metrics(image: np.ndarray) -> dict[str, float]:
    smooth = gaussian_filter(image, sigma=1.0, mode="constant", cval=0.0)
    gy, gx = np.gradient(smooth)
    laplacian = (
        np.roll(smooth, 1, axis=0)
        + np.roll(smooth, -1, axis=0)
        + np.roll(smooth, 1, axis=1)
        + np.roll(smooth, -1, axis=1)
        - 4.0 * smooth
    )
    return {
        "variance": float(np.var(smooth)),
        "gradient_energy": float(np.mean(gx * gx + gy * gy)),
        "laplacian_energy": float(np.mean(laplacian * laplacian)),
    }


def iter_candidates(search_radius: int, step: int):
    values = range(-search_radius, search_radius + 1, step)
    return itertools.product(values, repeat=2)
