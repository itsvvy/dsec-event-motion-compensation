from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsec_mc.blockwise import BlockwiseTranslation
from dsec_mc.homography import (
    _homography_displacement,
    estimate_homography_from_event_groups,
    estimate_local_homography_field,
)


def _synthetic_homography() -> np.ndarray:
    return np.asarray(
        [
            [1.02, 0.025, 2.5],
            [-0.018, 0.985, -1.5],
            [2.0e-5, -1.5e-5, 1.0],
        ],
        dtype=np.float64,
    )


def test_event_image_ecc_recovers_homography() -> None:
    import cv2
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(23)
    base = gaussian_filter(rng.random((128, 128)).astype(np.float32), sigma=1.3)
    base = (base - base.min()) / (base.max() - base.min())
    true_homography = _synthetic_homography()
    moving = cv2.warpPerspective(
        base,
        true_homography,
        (128, 128),
        flags=cv2.INTER_LINEAR,
    )
    early = np.repeat(base[None, :, :], 3, axis=0)
    late = np.repeat(moving[None, :, :], 3, axis=0)
    time_delta = 2.0 / 3.0
    yy, xx = np.indices(base.shape, dtype=np.float32)
    true_vx, true_vy, _ = _homography_displacement(
        true_homography,
        xx,
        yy,
        time_delta=time_delta,
    )
    initial_endpoint = (float(true_vx[64, 64]), float(true_vy[64, 64]))
    estimated, correlation = estimate_homography_from_event_groups(
        early,
        late,
        time_delta=time_delta,
        initial_endpoint=initial_endpoint,
    )
    assert estimated is not None
    assert correlation > 0.8
    estimated_vx, estimated_vy, valid = _homography_displacement(
        estimated,
        xx,
        yy,
        time_delta=time_delta,
    )
    assert np.all(valid)
    assert np.mean(np.hypot(estimated_vx - true_vx, estimated_vy - true_vy)) < 0.8


def test_local_homography_field_uses_blockwise_fallback() -> None:
    import cv2
    from scipy.ndimage import gaussian_filter

    size = 128
    rng = np.random.default_rng(29)
    base = gaussian_filter(rng.random((size, size)).astype(np.float32), sigma=1.3)
    base = (base - base.min()) / (base.max() - base.min())
    true_homography = _synthetic_homography()
    moving = cv2.warpPerspective(
        base,
        true_homography,
        (size, size),
        flags=cv2.INTER_LINEAR,
    )
    volume = np.repeat(base[None, :, :], 9, axis=0)
    volume[-3:] = moving[None, :, :]

    block_vx = np.asarray([[2.0]], dtype=np.float32)
    block_vy = np.asarray([[0.0]], dtype=np.float32)
    blockwise = BlockwiseTranslation(
        vx_grid=block_vx,
        vy_grid=block_vy,
        initial_vx_grid=block_vx.copy(),
        initial_vy_grid=block_vy.copy(),
        confidence_grid=np.ones_like(block_vx),
        valid_grid=np.ones_like(block_vx, dtype=bool),
        source_grid=np.full(block_vx.shape, "synthetic", dtype="<U22"),
        center_y=np.asarray([(size - 1) / 2], dtype=np.float32),
        center_x=np.asarray([(size - 1) / 2], dtype=np.float32),
        fallback_vx=2.0,
        fallback_vy=0.0,
    )
    base_vx = np.full((size, size), 2.0, dtype=np.float32)
    base_vy = np.zeros((size, size), dtype=np.float32)
    field = estimate_local_homography_field(
        volume,
        blockwise,
        base_vx_field=base_vx,
        base_vy_field=base_vy,
        height=size,
        width=size,
        region_size=size,
        stride=size,
        min_correlation=0.5,
        min_contrast_gain=1.0,
        max_local_variation=64.0,
    )
    assert np.count_nonzero(field.source_grid == "ecc-homography") == 1
    yy, xx = np.indices((size, size), dtype=np.float32)
    true_vx, true_vy, _ = _homography_displacement(
        true_homography,
        xx,
        yy,
        time_delta=2.0 / 3.0,
    )
    central = (slice(32, 96), slice(32, 96))
    error = np.hypot(
        field.dense_vx[central] - true_vx[central],
        field.dense_vy[central] - true_vy[central],
    )
    assert np.median(error) < 2.0


def test_rejected_homography_regions_keep_blockwise_field() -> None:
    size = 96
    block_vx = np.asarray([[4.0]], dtype=np.float32)
    block_vy = np.asarray([[-2.0]], dtype=np.float32)
    blockwise = BlockwiseTranslation(
        vx_grid=block_vx,
        vy_grid=block_vy,
        initial_vx_grid=block_vx.copy(),
        initial_vy_grid=block_vy.copy(),
        confidence_grid=np.ones_like(block_vx),
        valid_grid=np.ones_like(block_vx, dtype=bool),
        source_grid=np.full(block_vx.shape, "synthetic", dtype="<U22"),
        center_y=np.asarray([(size - 1) / 2], dtype=np.float32),
        center_x=np.asarray([(size - 1) / 2], dtype=np.float32),
        fallback_vx=4.0,
        fallback_vy=-2.0,
    )
    base_vx = np.linspace(-4.0, 4.0, size, dtype=np.float32)[None, :].repeat(
        size,
        axis=0,
    )
    base_vy = np.linspace(2.0, -2.0, size, dtype=np.float32)[:, None].repeat(
        size,
        axis=1,
    )
    field = estimate_local_homography_field(
        np.zeros((9, size, size), dtype=np.float32),
        blockwise,
        base_vx_field=base_vx,
        base_vy_field=base_vy,
        height=size,
        width=size,
        region_size=size,
        stride=size,
    )
    assert np.count_nonzero(field.source_grid == "fallback") == 1
    np.testing.assert_array_equal(field.dense_vx, base_vx)
    np.testing.assert_array_equal(field.dense_vy, base_vy)
