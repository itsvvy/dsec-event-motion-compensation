from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsec_mc.algorithm import estimate_translation, slice_local_events


def test_recovers_known_constant_translation() -> None:
    rng = np.random.default_rng(4)
    radius = 36
    bins = 10
    true_vx = 9.0
    true_vy = -5.0
    size = 2 * radius + 1

    base = gaussian_filter(
        (rng.random((size, size)) > 0.985).astype(np.float32),
        sigma=0.8,
    )
    yy, xx = np.nonzero(base > 0.08)
    repeats = 30
    pattern_x = np.tile(xx, repeats)
    pattern_y = np.tile(yy, repeats)
    time = np.repeat((np.arange(repeats) + 0.5) / repeats, len(xx))
    polarity = np.where((pattern_x + pattern_y) % 2, 1, -1).astype(np.uint8)

    x = pattern_x + (time - 0.5) * true_vx
    y = pattern_y + (time - 0.5) * true_vy
    slices, _ = slice_local_events(
        x,
        y,
        polarity,
        time,
        center_y=radius,
        center_x=radius,
        radius=radius,
        bins=bins,
    )
    estimate = estimate_translation(
        slices,
        radius=radius,
        search_radius=15,
        coarse_step=3,
        fine_radius=3,
    )
    assert np.linalg.norm([estimate.vx - true_vx, estimate.vy - true_vy]) < 1.5
    assert estimate.compensated_score > estimate.baseline_score
