from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsec_mc.affine import (
    estimate_affine_from_event_groups,
    fit_local_affine_from_blocks,
)
from dsec_mc.blockwise import BlockwiseTranslation


def test_local_affine_recovers_linear_motion_field() -> None:
    width = 640
    height = 480
    center_y = np.arange(32.0, height, 64.0, dtype=np.float32)
    center_x = np.arange(32.0, width, 64.0, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(center_y, center_x, indexing="ij")
    u = 2.0 * grid_x / (width - 1) - 1.0
    v = 2.0 * grid_y / (height - 1) - 1.0

    coefficient_x = np.asarray([2.0, -3.0, 4.0])
    coefficient_y = np.asarray([3.0, 2.0, -1.0])
    vx = coefficient_x[0] * u + coefficient_x[1] * v + coefficient_x[2]
    vy = coefficient_y[0] * u + coefficient_y[1] * v + coefficient_y[2]
    blockwise = BlockwiseTranslation(
        vx_grid=vx.astype(np.float32),
        vy_grid=vy.astype(np.float32),
        initial_vx_grid=vx.astype(np.float32),
        initial_vy_grid=vy.astype(np.float32),
        confidence_grid=np.full_like(vx, 0.9, dtype=np.float32),
        valid_grid=np.ones_like(vx, dtype=bool),
        source_grid=np.full(vx.shape, "short-consensus", dtype="<U22"),
        center_y=center_y,
        center_x=center_x,
        fallback_vx=float(np.mean(vx)),
        fallback_vy=float(np.mean(vy)),
    )
    affine = fit_local_affine_from_blocks(
        blockwise,
        height=height,
        width=width,
        neighbourhood_radius=3,
    )
    estimated_vx, estimated_vy = affine.interpolate(height, width)
    yy, xx = np.indices((height, width))
    expected_vx = (
        coefficient_x[0] * (2.0 * xx / (width - 1) - 1.0)
        + coefficient_x[1] * (2.0 * yy / (height - 1) - 1.0)
        + coefficient_x[2]
    )
    expected_vy = (
        coefficient_y[0] * (2.0 * xx / (width - 1) - 1.0)
        + coefficient_y[1] * (2.0 * yy / (height - 1) - 1.0)
        + coefficient_y[2]
    )
    assert np.mean(np.hypot(estimated_vx - expected_vx, estimated_vy - expected_vy)) < 0.2
    assert np.mean(affine.source_grid == "affine") > 0.95


def test_event_image_ecc_recovers_affine() -> None:
    import cv2
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(19)
    base = gaussian_filter(rng.random((128, 128)).astype(np.float32), sigma=1.4)
    base = (base - base.min()) / (base.max() - base.min())
    true_affine = np.asarray(
        [
            [1.02, 0.025, 2.0],
            [-0.018, 0.99, -1.0],
        ],
        dtype=np.float32,
    )
    moving = cv2.warpAffine(
        base,
        true_affine,
        (128, 128),
        flags=cv2.INTER_LINEAR,
    )
    early = np.repeat(base[None, :, :], 3, axis=0)
    late = np.repeat(moving[None, :, :], 3, axis=0)
    time_delta = 2.0 / 3.0
    initial_endpoint = true_affine[:, 2] / time_delta
    coefficients, correlation = estimate_affine_from_event_groups(
        early,
        late,
        time_delta=time_delta,
        initial_endpoint=tuple(initial_endpoint),
    )
    assert coefficients is not None
    assert correlation > 0.8
    expected = np.concatenate(
        [
            ((true_affine[:, :2] - np.eye(2)) / time_delta)[:, :1],
            ((true_affine[:, :2] - np.eye(2)) / time_delta)[:, 1:2],
            (true_affine[:, 2:] / time_delta),
        ],
        axis=1,
    ).T
    assert np.mean(np.abs(coefficients - expected)) < 0.35
