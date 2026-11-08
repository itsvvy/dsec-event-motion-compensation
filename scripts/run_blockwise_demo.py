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

from dsec_mc.algorithm import estimate_translation, refocus_event_slices
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
    save_blockwise_benchmark,
    save_blockwise_figure,
)


PATCHES = (
    (168, 480),
    (168, 72),
    (240, 96),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/blockwise")
    parser.add_argument("--flow-indices", type=int, nargs="+", default=[2, 8, 12])
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--refine-radius", type=int, default=3)
    parser.add_argument("--global-search-radius", type=int, default=32)
    parser.add_argument("--block-max-shift", type=float, default=48.0)
    return parser.parse_args()


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
        f"block size={args.tile_size}, stride={args.stride}, "
        f"temporal bins={args.bins}"
    )
    results: list[dict] = []

    for flow_index in args.flow_indices:
        if not 0 <= flow_index < len(timestamps):
            raise ValueError(f"flow index must be in [0, {len(timestamps) - 1}].")
        t_start_us, t_end_us = map(int, timestamps[flow_index])
        flow_xy, flow_valid = load_flow_target(files["flow_zip"], flow_index)
        events = load_event_window(files["events_h5"], t_start_us, t_end_us)
        x, y = rectify_xy(events.x, events.y, rectify_map)
        window_started = time.perf_counter()
        volume = slice_event_images(
            x,
            y,
            events.polarity,
            events.normalized_time,
            width=width,
            height=height,
            bins=args.bins,
        )
        slices = [item for item in volume]
        global_estimate = estimate_translation(
            slices,
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
        vx_field, vy_field = block_estimate.interpolate(height, width)
        vx_field = np.clip(vx_field, -args.block_max_shift, args.block_max_shift)
        vy_field = np.clip(vy_field, -args.block_max_shift, args.block_max_shift)
        global_vx = np.full((height, width), global_estimate.vx, dtype=np.float32)
        global_vy = np.full((height, width), global_estimate.vy, dtype=np.float32)

        uncompensated, _ = warp_event_volume(
            volume,
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )
        globally_compensated, _ = warp_event_volume(volume, global_vx, global_vy)
        blockwise_compensated, coverage = warp_event_volume(volume, vx_field, vy_field)

        global_metrics = dense_motion_metrics(
            global_vx,
            global_vy,
            flow_xy,
            flow_valid,
        )
        block_metrics = dense_motion_metrics(
            vx_field,
            vy_field,
            flow_xy,
            flow_valid,
        )
        global_image_metrics = image_change_metrics(
            uncompensated,
            globally_compensated,
        )
        block_image_metrics = image_change_metrics(
            uncompensated,
            blockwise_compensated,
        )

        row = {
            "flow_index": flow_index,
            "flow_png": f"{2 * flow_index + 2:06d}.png",
            "window_ms": (t_end_us - t_start_us) / 1000.0,
            "event_count": int(len(events.t_us)),
            "valid_blocks": int(block_estimate.valid_grid.sum()),
            "total_blocks": int(block_estimate.valid_grid.size),
            "mean_block_confidence": float(block_estimate.confidence_grid.mean()),
            "warp_coverage": float(coverage.mean()),
            "global_vx_px": float(global_estimate.vx),
            "global_vy_px": float(global_estimate.vy),
            "block_mean_vx_px": float(vx_field.mean()),
            "block_mean_vy_px": float(vy_field.mean()),
            "global_median_endpoint_error_px": global_metrics[
                "median_endpoint_error_px"
            ],
            "blockwise_median_endpoint_error_px": block_metrics[
                "median_endpoint_error_px"
            ],
            "global_mean_endpoint_error_px": global_metrics[
                "mean_endpoint_error_px"
            ],
            "blockwise_mean_endpoint_error_px": block_metrics[
                "mean_endpoint_error_px"
            ],
            "global_within_2px": global_metrics["within_2px"],
            "blockwise_within_2px": block_metrics["within_2px"],
            "global_variance_gain": global_image_metrics["variance_gain"],
            "blockwise_variance_gain": block_image_metrics["variance_gain"],
            "global_gradient_energy_gain": global_image_metrics[
                "gradient_energy_gain"
            ],
            "blockwise_gradient_energy_gain": block_image_metrics[
                "gradient_energy_gain"
            ],
            "runtime_s": time.perf_counter() - window_started,
        }
        for source in (
            "short-consensus",
            "short-single",
            "long",
            "long-low-confidence",
            "fallback",
            "none",
        ):
            key = f"blocks_{source.replace('-', '_')}"
            row[key] = int(np.count_nonzero(block_estimate.source_grid == source))
        for center_y, center_x in PATCHES:
            radius = 64
            region = (
                max(0, center_y - radius),
                min(height, center_y + radius),
                max(0, center_x - radius),
                min(width, center_x + radius),
            )
            global_patch = dense_motion_metrics(
                global_vx,
                global_vy,
                flow_xy,
                flow_valid,
                region=region,
            )
            block_patch = dense_motion_metrics(
                vx_field,
                vy_field,
                flow_xy,
                flow_valid,
                region=region,
            )
            suffix = f"y{center_y}_x{center_x}"
            row[f"global_{suffix}_median_epe_px"] = global_patch[
                "median_endpoint_error_px"
            ]
            row[f"blockwise_{suffix}_median_epe_px"] = block_patch[
                "median_endpoint_error_px"
            ]
            patch_valid = flow_valid[region[0]:region[1], region[2]:region[3]]
            patch_field = np.stack(
                [
                    vx_field[region[0]:region[1], region[2]:region[3]],
                    vy_field[region[0]:region[1], region[2]:region[3]],
                ],
                axis=-1,
            )[patch_valid]
            row[f"blockwise_{suffix}_field_median_vx_px"] = float(
                np.median(patch_field[:, 0])
            )
            row[f"blockwise_{suffix}_field_median_vy_px"] = float(
                np.median(patch_field[:, 1])
            )
        results.append(row)

        figure_path = args.output_dir / f"blockwise_f{flow_index:02d}.png"
        save_blockwise_figure(
            uncompensated,
            globally_compensated,
            blockwise_compensated,
            vx_field=vx_field,
            vy_field=vy_field,
            flow_xy=flow_xy,
            valid=flow_valid,
            global_metrics=global_metrics,
            blockwise_metrics=block_metrics,
            title=f"DSEC {flow_index:02d}, {row['window_ms']:.1f} ms",
            output_path=figure_path,
        )
        print(
            f"window {flow_index:02d}: "
            f"global median EPE={global_metrics['median_endpoint_error_px']:.2f}px, "
            f"block median EPE={block_metrics['median_endpoint_error_px']:.2f}px, "
            f"coverage={coverage.mean() * 100:.1f}%, "
            f"valid blocks={block_estimate.valid_grid.sum()}/{block_estimate.valid_grid.size}, "
            f"time={row['runtime_s']:.2f}s"
        )

    benchmark_runtime = time.perf_counter() - benchmark_started
    summary = {
        "sequence": "thun_00_a",
        "flow_indices": args.flow_indices,
        "bins": args.bins,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "refine_radius": args.refine_radius,
        "data_prepare_runtime_s": prepare_time,
        "benchmark_runtime_s": benchmark_runtime,
        "results": results,
        "mean_global_median_epe_px": float(
            np.mean([item["global_median_endpoint_error_px"] for item in results])
        ),
        "mean_blockwise_median_epe_px": float(
            np.mean([item["blockwise_median_endpoint_error_px"] for item in results])
        ),
        "mean_global_within_2px": float(
            np.mean([item["global_within_2px"] for item in results])
        ),
        "mean_blockwise_within_2px": float(
            np.mean([item["blockwise_within_2px"] for item in results])
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    save_blockwise_benchmark(
        results,
        output_path=args.output_dir / "benchmark_summary.png",
    )
    print(f"\nSaved: {args.output_dir / 'summary.json'}")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
