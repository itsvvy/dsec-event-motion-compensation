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

from dsec_mc.algorithm import estimate_translation, refocus_event_slices, slice_local_events
from dsec_mc.data import (
    load_event_window,
    load_flow_target,
    load_flow_timestamps,
    load_rectification_map,
    prepare_dsec_files,
    rectify_xy,
)
from dsec_mc.evaluation import (
    compare_motion,
    ground_truth_translation,
    image_change_metrics,
    save_benchmark_summary,
    save_qualitative_figure,
)


PATCHES = (
    (168, 480),
    (168, 72),
    (240, 96),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--flow-indices",
        type=int,
        nargs="+",
        default=[2, 8, 12],
        help="DSEC forward-flow frame indices to evaluate.",
    )
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument("--coarse-step", type=int, default=4)
    parser.add_argument(
        "--patches",
        type=int,
        nargs="+",
        action="append",
        default=None,
        help="Optional patch centers as Y X pairs after each --patches flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    files = prepare_dsec_files(args.data_dir)
    data_prepare_runtime_s = time.perf_counter() - started
    benchmark_started = time.perf_counter()
    timestamps = load_flow_timestamps(files["flow_times"])
    rectify_map = load_rectification_map(files["rectify_h5"])

    if args.patches:
        patches = []
        for pair in args.patches:
            if len(pair) != 2:
                raise ValueError("--patches expects exactly two integers (Y X).")
            patches.append(tuple(pair))
    else:
        patches = PATCHES

    results = []
    print(f"CPU: {platform.processor() or platform.machine()}")

    for flow_index in args.flow_indices:
        if not 0 <= flow_index < len(timestamps):
            raise ValueError(f"flow index must be in [0, {len(timestamps) - 1}].")
        t_start_us, t_end_us = map(int, timestamps[flow_index])
        flow_xy, valid = load_flow_target(files["flow_zip"], flow_index)
        events = load_event_window(files["events_h5"], t_start_us, t_end_us)
        rectified_x, rectified_y = rectify_xy(events.x, events.y, rectify_map)
        print(
            f"\nDSEC window {flow_index}: {t_start_us}..{t_end_us} us, "
            f"{len(events.t_us):,} raw events, {len(patches)} patches"
        )

        for center_y, center_x in patches:
            patch_started = time.perf_counter()
            gt_xy, valid_count = ground_truth_translation(
                flow_xy,
                valid,
                center_y=center_y,
                center_x=center_x,
                radius=args.radius,
            )
            slices, _bounds = slice_local_events(
                rectified_x,
                rectified_y,
                events.polarity,
                events.normalized_time,
                center_y=center_y,
                center_x=center_x,
                radius=args.radius,
                bins=args.bins,
            )
            local_event_count = int(sum(np.count_nonzero(item) for item in slices))
            if local_event_count < 1000:
                print(
                    f"Skip patch ({center_y}, {center_x}): "
                    f"only {local_event_count} events."
                )
                continue

            estimate = estimate_translation(
                slices,
                radius=args.radius,
                search_radius=args.search_radius,
                coarse_step=args.coarse_step,
                sigma=1.0,
            )
            baseline = refocus_event_slices(
                slices,
                (0.0, 0.0),
                max_shift=max(args.search_radius, 4),
            )
            compensated = refocus_event_slices(
                slices,
                (estimate.vx, estimate.vy),
                max_shift=max(args.search_radius, 4),
            )
            motion_metrics = compare_motion(estimate, gt_xy)
            image_metrics = image_change_metrics(baseline, compensated)
            result = {
                "flow_index": flow_index,
                "flow_png": f"{2 * flow_index + 2:06d}.png",
                "time_start_us": t_start_us,
                "time_end_us": t_end_us,
                "window_ms": (t_end_us - t_start_us) / 1000.0,
                "patch_center_yx": [center_y, center_x],
                "patch_event_count": local_event_count,
                "raw_event_count": len(events.t_us),
                "flow_valid_count": valid_count,
                "runtime_s": time.perf_counter() - patch_started,
                **motion_metrics,
                **image_metrics,
            }
            results.append(result)

            figure_path = (
                args.output_dir
                / f"dsec_f{flow_index:02d}_y{center_y}_x{center_x}.png"
            )
            save_qualitative_figure(
                baseline,
                compensated,
                estimate,
                patch_center=(center_y, center_x),
                title=f"DSEC {flow_index:02d}, {result['window_ms']:.1f} ms",
                output_path=figure_path,
            )
            print(
                f"patch ({center_y:3d},{center_x:3d}) events={local_event_count:6d}: "
                f"v_est=({estimate.vx:6.2f},{estimate.vy:6.2f}) "
                f"v_gt=({gt_xy[0]:6.2f},{gt_xy[1]:6.2f}) "
                f"EPE={result['endpoint_error_px']:.2f}px "
                f"variance_gain={result['variance_gain']:.2f}x "
                f"time={result['runtime_s']:.1f}s"
            )

    summary = {
        "sequence": "thun_00_a",
        "flow_indices": args.flow_indices,
        "bins": args.bins,
        "patch_radius_px": args.radius,
        "search_radius_px": args.search_radius,
        "total_runtime_s": time.perf_counter() - started,
        "data_prepare_runtime_s": data_prepare_runtime_s,
        "benchmark_runtime_s": time.perf_counter() - benchmark_started,
        "results": results,
    }
    if results:
        summary["mean_endpoint_error_px"] = float(
            np.mean([item["endpoint_error_px"] for item in results])
        )
        summary["median_endpoint_error_px"] = float(
            np.median([item["endpoint_error_px"] for item in results])
        )
        summary["median_variance_gain"] = float(
            np.median([item["variance_gain"] for item in results])
        )
        summary["mean_gradient_energy_gain"] = float(
            np.mean([item["gradient_energy_gain"] for item in results])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if results:
        csv_path = args.output_dir / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(results[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(results)
        save_benchmark_summary(
            results,
            output_path=args.output_dir / "benchmark_summary.png",
        )

    print(f"\nSaved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
