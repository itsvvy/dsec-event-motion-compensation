from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dsec_mc.affine import estimate_local_affine_field
from dsec_mc.algorithm import estimate_translation
from dsec_mc.blockwise import (
    estimate_blockwise_translation,
    slice_event_images,
    warp_event_volume,
)
from dsec_mc.data import (
    load_event_window,
    load_flow_target,
    load_flow_timestamps,
    load_rectification_map,
    prepare_dsec_files,
    rectify_xy,
)
from dsec_mc.evaluation import (
    dense_motion_metrics,
    image_change_metrics,
    save_three_model_benchmark,
    save_three_model_figure,
)
from dsec_mc.homography import estimate_local_homography_field


PATCHES = (
    (168, 480),
    (168, 72),
    (240, 96),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/homography",
    )
    parser.add_argument("--flow-indices", type=int, nargs="+", default=[2, 8, 12])
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--refine-radius", type=int, default=3)
    parser.add_argument("--global-search-radius", type=int, default=32)
    parser.add_argument("--block-max-shift", type=float, default=48.0)
    parser.add_argument("--affine-region-size", type=int, default=160)
    parser.add_argument("--affine-stride", type=int, default=80)
    parser.add_argument("--affine-min-correlation", type=float, default=0.30)
    parser.add_argument("--affine-min-contrast-gain", type=float, default=1.02)
    parser.add_argument("--affine-max-variation", type=float, default=24.0)
    parser.add_argument("--homography-region-size", type=int, default=160)
    parser.add_argument("--homography-stride", type=int, default=80)
    parser.add_argument("--homography-min-correlation", type=float, default=0.30)
    parser.add_argument("--homography-min-contrast-gain", type=float, default=1.02)
    parser.add_argument("--homography-max-variation", type=float, default=32.0)
    parser.add_argument("--homography-min-denominator", type=float, default=0.10)
    return parser.parse_args()


def _field_metrics(
    vx: np.ndarray,
    vy: np.ndarray,
    flow_xy: np.ndarray,
    flow_valid: np.ndarray,
) -> dict[str, float | int]:
    return dense_motion_metrics(vx, vy, flow_xy, flow_valid)


def _median_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    files = prepare_dsec_files(args.data_dir)
    prepare_time = time.perf_counter() - started
    benchmark_started = time.perf_counter()
    timestamps = load_flow_timestamps(files["flow_times"])
    rectify_map = load_rectification_map(files["rectify_h5"])
    height, width = rectify_map.shape[:2]
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(
        f"B: block={args.tile_size}, stride={args.stride} | "
        f"BA/BH regions={args.affine_region_size}, stride={args.affine_stride}"
    )

    results: list[dict] = []
    for flow_index in args.flow_indices:
        if not 0 <= flow_index < len(timestamps):
            raise ValueError(f"flow index must be in [0, {len(timestamps) - 1}].")
        window_started = time.perf_counter()
        t_start_us, t_end_us = map(int, timestamps[flow_index])
        flow_xy, flow_valid = load_flow_target(files["flow_zip"], flow_index)
        events = load_event_window(files["events_h5"], t_start_us, t_end_us)
        x, y = rectify_xy(events.x, events.y, rectify_map)
        volume = slice_event_images(
            x,
            y,
            events.polarity,
            events.normalized_time,
            width=width,
            height=height,
            bins=args.bins,
        )

        block_started = time.perf_counter()
        global_estimate = estimate_translation(
            [item for item in volume],
            radius=min(height, width) // 2,
            search_radius=args.global_search_radius,
            coarse_step=4,
            fine_radius=4,
        )
        block_estimate = estimate_blockwise_translation(
            volume,
            tile_size=args.tile_size,
            stride=args.stride,
            phase_bins=3,
            refine_radius=args.refine_radius,
            max_abs_shift=args.block_max_shift,
            fallback=(global_estimate.vx, global_estimate.vy),
        )
        block_vx, block_vy = block_estimate.interpolate(height, width)
        block_vx = np.clip(block_vx, -args.block_max_shift, args.block_max_shift)
        block_vy = np.clip(block_vy, -args.block_max_shift, args.block_max_shift)
        block_runtime = time.perf_counter() - block_started

        affine_started = time.perf_counter()
        affine = estimate_local_affine_field(
            volume,
            block_estimate,
            base_vx_field=block_vx,
            base_vy_field=block_vy,
            height=height,
            width=width,
            region_size=args.affine_region_size,
            stride=args.affine_stride,
            min_correlation=args.affine_min_correlation,
            min_contrast_gain=args.affine_min_contrast_gain,
            max_local_variation=args.affine_max_variation,
            max_abs_shift=args.block_max_shift,
        )
        affine_vx, affine_vy = affine.interpolate(height, width)
        affine_vx = np.clip(affine_vx, -args.block_max_shift, args.block_max_shift)
        affine_vy = np.clip(affine_vy, -args.block_max_shift, args.block_max_shift)
        affine_runtime = time.perf_counter() - affine_started

        homography_started = time.perf_counter()
        homography = estimate_local_homography_field(
            volume,
            block_estimate,
            base_vx_field=block_vx,
            base_vy_field=block_vy,
            height=height,
            width=width,
            region_size=args.homography_region_size,
            stride=args.homography_stride,
            min_correlation=args.homography_min_correlation,
            min_contrast_gain=args.homography_min_contrast_gain,
            max_local_variation=args.homography_max_variation,
            max_abs_shift=args.block_max_shift,
            min_denominator=args.homography_min_denominator,
        )
        homography_vx, homography_vy = homography.interpolate(height, width)
        homography_vx = np.clip(
            homography_vx,
            -args.block_max_shift,
            args.block_max_shift,
        )
        homography_vy = np.clip(
            homography_vy,
            -args.block_max_shift,
            args.block_max_shift,
        )
        homography_runtime = time.perf_counter() - homography_started

        warp_started = time.perf_counter()
        uncompensated, _ = warp_event_volume(
            volume,
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )
        block_image, block_coverage = warp_event_volume(volume, block_vx, block_vy)
        affine_image, affine_coverage = warp_event_volume(volume, affine_vx, affine_vy)
        homography_image, homography_coverage = warp_event_volume(
            volume,
            homography_vx,
            homography_vy,
        )
        warp_runtime = time.perf_counter() - warp_started

        block_metrics = _field_metrics(block_vx, block_vy, flow_xy, flow_valid)
        affine_metrics = _field_metrics(affine_vx, affine_vy, flow_xy, flow_valid)
        homography_metrics = _field_metrics(
            homography_vx,
            homography_vy,
            flow_xy,
            flow_valid,
        )
        block_image_metrics = image_change_metrics(uncompensated, block_image)
        affine_image_metrics = image_change_metrics(uncompensated, affine_image)
        homography_image_metrics = image_change_metrics(
            uncompensated,
            homography_image,
        )
        row = {
            "flow_index": flow_index,
            "window_ms": (t_end_us - t_start_us) / 1000.0,
            "event_count": int(len(events.t_us)),
            "blockwise_median_endpoint_error_px": block_metrics[
                "median_endpoint_error_px"
            ],
            "affine_median_endpoint_error_px": affine_metrics[
                "median_endpoint_error_px"
            ],
            "homography_median_endpoint_error_px": homography_metrics[
                "median_endpoint_error_px"
            ],
            "blockwise_mean_endpoint_error_px": block_metrics[
                "mean_endpoint_error_px"
            ],
            "affine_mean_endpoint_error_px": affine_metrics["mean_endpoint_error_px"],
            "homography_mean_endpoint_error_px": homography_metrics[
                "mean_endpoint_error_px"
            ],
            "blockwise_p90_endpoint_error_px": block_metrics[
                "p90_endpoint_error_px"
            ],
            "affine_p90_endpoint_error_px": affine_metrics["p90_endpoint_error_px"],
            "homography_p90_endpoint_error_px": homography_metrics[
                "p90_endpoint_error_px"
            ],
            "blockwise_within_2px": block_metrics["within_2px"],
            "affine_within_2px": affine_metrics["within_2px"],
            "homography_within_2px": homography_metrics["within_2px"],
            "blockwise_variance_gain": block_image_metrics["variance_gain"],
            "affine_variance_gain": affine_image_metrics["variance_gain"],
            "homography_variance_gain": homography_image_metrics["variance_gain"],
            "blockwise_gradient_energy_gain": block_image_metrics[
                "gradient_energy_gain"
            ],
            "affine_gradient_energy_gain": affine_image_metrics[
                "gradient_energy_gain"
            ],
            "homography_gradient_energy_gain": homography_image_metrics[
                "gradient_energy_gain"
            ],
            "blockwise_warp_coverage": float(block_coverage.mean()),
            "affine_warp_coverage": float(affine_coverage.mean()),
            "homography_warp_coverage": float(homography_coverage.mean()),
            "affine_regions": int(np.count_nonzero(affine.source_grid == "ecc-affine")),
            "affine_fallback_regions": int(
                np.count_nonzero(affine.source_grid == "fallback")
            ),
            "homography_regions": int(
                np.count_nonzero(homography.source_grid == "ecc-homography")
            ),
            "homography_fallback_regions": int(
                np.count_nonzero(homography.source_grid == "fallback")
            ),
            "median_affine_correlation": _median_or_nan(affine.correlation_grid),
            "median_homography_correlation": _median_or_nan(
                homography.correlation_grid
            ),
            "block_runtime_s": block_runtime,
            "affine_runtime_s": affine_runtime,
            "homography_runtime_s": homography_runtime,
            "warp_runtime_s": warp_runtime,
            "window_runtime_s": time.perf_counter() - window_started,
        }
        for center_y, center_x in PATCHES:
            radius = 64
            region = (
                max(0, center_y - radius),
                min(height, center_y + radius),
                max(0, center_x - radius),
                min(width, center_x + radius),
            )
            suffix = f"y{center_y}_x{center_x}"
            for prefix, vx_field, vy_field in (
                ("blockwise", block_vx, block_vy),
                ("affine", affine_vx, affine_vy),
                ("homography", homography_vx, homography_vy),
            ):
                patch_metrics = dense_motion_metrics(
                    vx_field,
                    vy_field,
                    flow_xy,
                    flow_valid,
                    region=region,
                )
                row[f"{prefix}_{suffix}_median_epe_px"] = patch_metrics[
                    "median_endpoint_error_px"
                ]
        results.append(row)

        save_three_model_figure(
            uncompensated,
            block_image,
            affine_image,
            homography_image,
            affine_vx=affine_vx,
            affine_vy=affine_vy,
            homography_vx=homography_vx,
            homography_vy=homography_vy,
            metrics={
                "blockwise": block_metrics,
                "affine": affine_metrics,
                "homography": homography_metrics,
            },
            title=f"DSEC {flow_index:02d}, {row['window_ms']:.1f} ms",
            output_path=args.output_dir / f"homography_f{flow_index:02d}.png",
        )
        print(
            f"window {flow_index:02d}: "
            f"B={block_metrics['median_endpoint_error_px']:.2f}px, "
            f"BA={affine_metrics['median_endpoint_error_px']:.2f}px, "
            f"BH={homography_metrics['median_endpoint_error_px']:.2f}px | "
            f"BH regions={row['homography_regions']}/"
            f"{homography.source_grid.size}, "
            f"time={row['window_runtime_s']:.2f}s"
        )

    runtime = time.perf_counter() - benchmark_started
    summary = {
        "sequence": "thun_00_a",
        "flow_indices": args.flow_indices,
        "bins": args.bins,
        "block_tile_size": args.tile_size,
        "block_stride": args.stride,
        "affine_region_size": args.affine_region_size,
        "affine_stride": args.affine_stride,
        "homography_region_size": args.homography_region_size,
        "homography_stride": args.homography_stride,
        "homography_min_correlation": args.homography_min_correlation,
        "homography_min_contrast_gain": args.homography_min_contrast_gain,
        "homography_max_variation_px": args.homography_max_variation,
        "homography_min_denominator": args.homography_min_denominator,
        "data_prepare_runtime_s": prepare_time,
        "benchmark_runtime_s": runtime,
        "results": results,
    }
    for prefix in ("blockwise", "affine", "homography"):
        summary[f"mean_{prefix}_median_epe_px"] = float(
            np.mean(
                [item[f"{prefix}_median_endpoint_error_px"] for item in results]
            )
        )
        summary[f"mean_{prefix}_within_2px"] = float(
            np.mean([item[f"{prefix}_within_2px"] for item in results])
        )
        summary[f"mean_{prefix}_variance_gain"] = float(
            np.mean([item[f"{prefix}_variance_gain"] for item in results])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    save_three_model_benchmark(
        results,
        output_path=args.output_dir / "benchmark_summary.png",
    )
    print(f"\nSaved: {args.output_dir / 'summary.json'}")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
