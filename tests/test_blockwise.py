from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import shift as image_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsec_mc.blockwise import (
    choose_phase_initialization,
    estimate_blockwise_translation,
    phase_correlation_shift,
    slice_event_images,
)


def test_short_baseline_resolves_spurious_zero_long_peak() -> None:
    vx, vy, source = choose_phase_initialization(
        long_candidate=(0.0, 0.0, 10.0),
        short_candidates=[
            (-18.1, 4.5, 12.0),
            (-17.5, 4.2, 13.0),
        ],
    )
    assert source == "short-consensus"
    assert abs(vx + 17.8) < 0.5
    assert abs(vy - 4.35) < 0.5


def test_phase_correlation_recovers_content_motion() -> None:
    rng = np.random.default_rng(7)
    reference = (rng.random((64, 64)) > 0.97).astype(np.float32)
    moving = image_shift(
        reference,
        shift=(-2.0, 3.0),
        order=1,
        mode="constant",
        cval=0.0,
    )
    dx, dy, confidence = phase_correlation_shift(reference, moving)
    assert abs(dx - 3.0) < 0.5
    assert abs(dy + 2.0) < 0.5
    assert confidence > 10.0


def test_blockwise_estimation_recovers_two_motion_regions() -> None:
    rng = np.random.default_rng(11)
    height = width = 128
    y, x = np.indices((height, width))
    texture = (rng.random((height, width)) > 0.982).astype(np.uint8)

    xs = []
    ys = []
    times = []
    polarities = []
    for tau in np.linspace(0.05, 0.95, 18):
        event_y, event_x = np.nonzero(texture)
        motion_x = np.where(event_x < width // 2, 10.0, -8.0)
        motion_y = np.where(event_x < width // 2, -3.0, 4.0)
        xs.append(event_x + motion_x * (tau - 0.5))
        ys.append(event_y + motion_y * (tau - 0.5))
        times.append(np.full(event_x.size, tau))
        polarities.append((event_y + event_x) % 2)

    volume = slice_event_images(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(polarities).astype(np.uint8),
        np.concatenate(times),
        width=width,
        height=height,
        bins=9,
    )
    estimate = estimate_blockwise_translation(
        volume,
        tile_size=64,
        stride=32,
        refine_radius=2,
        max_abs_shift=18.0,
        fallback=(0.0, 0.0),
    )
    left_column = int(np.argmin(np.abs(estimate.center_x - 32)))
    right_column = int(np.argmin(np.abs(estimate.center_x - 96)))
    middle_row = int(np.argmin(np.abs(estimate.center_y - 64)))
    left = np.asarray(
        [estimate.vx_grid[middle_row, left_column], estimate.vy_grid[middle_row, left_column]]
    )
    right = np.asarray(
        [estimate.vx_grid[middle_row, right_column], estimate.vy_grid[middle_row, right_column]]
    )
    assert np.linalg.norm(left - np.asarray([10.0, -3.0])) < 2.5
    assert np.linalg.norm(right - np.asarray([-8.0, 4.0])) < 2.5
