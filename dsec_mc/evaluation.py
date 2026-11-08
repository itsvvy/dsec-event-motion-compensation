from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .algorithm import MotionEstimate, sharpness_metrics


def ground_truth_translation(
    flow_xy: np.ndarray,
    valid: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> tuple[np.ndarray, int]:
    y0 = center_y - radius
    y1 = center_y + radius + 1
    x0 = center_x - radius
    x1 = center_x + radius + 1
    patch_flow = flow_xy[y0:y1, x0:x1]
    patch_valid = valid[y0:y1, x0:x1]
    values = patch_flow[patch_valid]
    if len(values) == 0:
        raise ValueError("No valid optical-flow vectors in the evaluation patch.")
    return np.median(values, axis=0), int(len(values))


def compare_motion(
    estimate: MotionEstimate,
    gt_xy: np.ndarray,
) -> dict[str, float]:
    estimated = np.asarray([estimate.vx, estimate.vy], dtype=np.float64)
    truth = np.asarray(gt_xy, dtype=np.float64)
    error = estimated - truth
    denominator = np.linalg.norm(estimated) * np.linalg.norm(truth)
    cosine = float(np.dot(estimated, truth) / denominator) if denominator > 1e-12 else 0.0
    return {
        "estimated_vx_px": float(estimated[0]),
        "estimated_vy_px": float(estimated[1]),
        "gt_vx_px": float(truth[0]),
        "gt_vy_px": float(truth[1]),
        "endpoint_error_px": float(np.linalg.norm(error)),
        "direction_error_deg": float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        ),
    }


def dense_motion_metrics(
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    flow_xy: np.ndarray,
    valid: np.ndarray,
    *,
    region: tuple[int, int, int, int] | None = None,
) -> dict[str, float | int]:
    """Compare endpoint displacement with DSEC forward flow at valid pixels."""
    if region is not None:
        y0, y1, x0, x1 = region
    else:
        y0, y1, x0, x1 = 0, flow_xy.shape[0], 0, flow_xy.shape[1]
    selected = valid[y0:y1, x0:x1]
    if not np.any(selected):
        raise ValueError("No valid DSEC flow vectors in the evaluation region.")
    predicted = np.stack(
        [vx_field[y0:y1, x0:x1], vy_field[y0:y1, x0:x1]],
        axis=-1,
    )[selected]
    truth = flow_xy[y0:y1, x0:x1][selected]
    error = np.linalg.norm(predicted - truth, axis=1)
    return {
        "valid_pixels": int(error.size),
        "mean_endpoint_error_px": float(np.mean(error)),
        "median_endpoint_error_px": float(np.median(error)),
        "p90_endpoint_error_px": float(np.percentile(error, 90)),
        "within_1px": float(np.mean(error <= 1.0)),
        "within_2px": float(np.mean(error <= 2.0)),
        "within_3px": float(np.mean(error <= 3.0)),
    }


def _normalise_for_display(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, sigma=1.0, mode="constant", cval=0.0)
    scale = np.percentile(np.abs(smooth), 99.5)
    if scale <= 1e-12:
        scale = 1.0
    return np.clip(0.5 + 0.5 * smooth / scale, 0.0, 1.0)


def save_qualitative_figure(
    baseline: np.ndarray,
    compensated: np.ndarray,
    estimate: MotionEstimate,
    *,
    patch_center: tuple[int, int],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_for_display(baseline), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Uncompensated event accumulation")
    axes[1].imshow(_normalise_for_display(compensated), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Motion-compensated at t_ref")

    extent = [
        float(estimate.search_vx[0]),
        float(estimate.search_vx[-1]),
        float(estimate.search_vy[-1]),
        float(estimate.search_vy[0]),
    ]
    image = axes[2].imshow(
        estimate.coarse_grid,
        cmap="magma",
        extent=extent,
        aspect="auto",
    )
    axes[2].scatter([estimate.vx], [estimate.vy], marker="x", c="cyan", s=70)
    axes[2].set_xlabel(r"$v_x$ (pixels / window)")
    axes[2].set_ylabel(r"$v_y$ (pixels / window)")
    axes[2].set_title("Variance objective")
    fig.colorbar(image, ax=axes[2], shrink=0.82, label="variance")

    for axis in axes[:2]:
        axis.axis("off")
    gain = estimate.compensated_score / max(estimate.baseline_score, 1e-12)
    fig.suptitle(
        f"{title} | patch y,x={patch_center} | "
        f"v=({estimate.vx:.2f}, {estimate.vy:.2f}) px | "
        f"contrast gain={gain:.2f}x",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def image_change_metrics(
    baseline: np.ndarray,
    compensated: np.ndarray,
) -> dict[str, float]:
    before = sharpness_metrics(baseline)
    after = sharpness_metrics(compensated)
    output = {
        f"baseline_{name}": value for name, value in before.items()
    }
    output.update({f"compensated_{name}": value for name, value in after.items()})
    for name in before:
        output[f"{name}_gain"] = after[name] / max(before[name], 1e-12)
    return output


def save_benchmark_summary(
    results: list[dict],
    *,
    output_path: Path,
) -> None:
    patch_labels = sorted(
        {tuple(item["patch_center_yx"]) for item in results}
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(patch_labels), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for color, patch in zip(colors, patch_labels):
        selected = [
            item for item in results if tuple(item["patch_center_yx"]) == patch
        ]
        x = [item["flow_index"] for item in selected]
        label = f"y={patch[0]}, x={patch[1]}"
        axes[0].plot(
            x,
            [item["endpoint_error_px"] for item in selected],
            marker="o",
            linewidth=1.7,
            color=color,
            label=label,
        )
        axes[1].plot(
            x,
            [item["variance_gain"] for item in selected],
            marker="o",
            linewidth=1.7,
            color=color,
            label=label,
        )

    axes[0].axhline(2.0, color="0.55", linestyle="--", linewidth=1, label="2 px")
    axes[0].set_title("Motion estimate error vs. DSEC flow")
    axes[0].set_ylabel("endpoint error (px)")
    axes[1].set_title("Event-image contrast improvement")
    axes[1].set_ylabel("variance gain (x)")
    for axis in axes:
        axis.set_xlabel("DSEC forward-flow window index")
        axis.set_xticks(sorted({item["flow_index"] for item in results}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_blockwise_figure(
    uncompensated: np.ndarray,
    global_compensated: np.ndarray,
    blockwise_compensated: np.ndarray,
    *,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    flow_xy: np.ndarray,
    valid: np.ndarray,
    global_metrics: dict,
    blockwise_metrics: dict,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = (
        (uncompensated, "Uncompensated events"),
        (global_compensated, "Global translation"),
        (blockwise_compensated, "Block-wise translation"),
    )
    for axis, (image, label) in zip(axes.flat[:3], panels):
        axis.imshow(_normalise_for_display(image), cmap="gray", vmin=0, vmax=1)
        axis.set_title(label)
        axis.axis("off")

    field_axis = axes[1, 1]
    magnitude = np.hypot(vx_field, vy_field)
    image = field_axis.imshow(magnitude, cmap="viridis")
    step = 12
    yy, xx = np.mgrid[0:vx_field.shape[0]:step, 0:vx_field.shape[1]:step]
    field_axis.quiver(
        xx,
        yy,
        vx_field[::step, ::step],
        vy_field[::step, ::step],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.2,
        width=0.0025,
    )
    field_axis.set_title("Estimated endpoint displacement")
    field_axis.set_xlim(0, vx_field.shape[1] - 1)
    field_axis.set_ylim(vx_field.shape[0] - 1, 0)
    fig.colorbar(image, ax=field_axis, shrink=0.82, label="pixels / window")

    fig.suptitle(
        f"{title}\n"
        f"global median EPE {global_metrics['median_endpoint_error_px']:.2f} px | "
        f"blockwise median EPE {blockwise_metrics['median_endpoint_error_px']:.2f} px | "
        f"DSEC-valid coverage {np.mean(valid) * 100:.1f}%",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_blockwise_benchmark(
    results: list[dict],
    *,
    output_path: Path,
) -> None:
    x = np.arange(len(results), dtype=np.float32)
    labels = [str(item["flow_index"]) for item in results]
    global_epe = [item["global_median_endpoint_error_px"] for item in results]
    block_epe = [item["blockwise_median_endpoint_error_px"] for item in results]
    global_2px = [100.0 * item["global_within_2px"] for item in results]
    block_2px = [100.0 * item["blockwise_within_2px"] for item in results]
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(x - width / 2, global_epe, width, label="global translation")
    axes[0].bar(x + width / 2, block_epe, width, label="block-wise")
    axes[0].set_ylabel("median EPE (px)")
    axes[0].set_title("DSEC flow error on valid pixels")

    axes[1].bar(x - width / 2, global_2px, width, label="global translation")
    axes[1].bar(x + width / 2, block_2px, width, label="block-wise")
    axes[1].set_ylabel("valid pixels with EPE <= 2 px (%)")
    axes[1].set_title("Motion accuracy coverage")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("DSEC forward-flow window")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
