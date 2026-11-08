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
