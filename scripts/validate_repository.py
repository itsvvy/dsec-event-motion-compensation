from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "notebooks").glob("*.ipynb")))
SUMMARY_PREFIXES = {
    "blockwise": ("blockwise",),
    "local_affine": ("blockwise", "affine"),
    "homography": ("blockwise", "affine", "homography"),
}
REQUIRED_IMAGES = (
    "outputs/benchmark_summary.png",
    "outputs/blockwise/benchmark_summary.png",
    "outputs/blockwise/blockwise_f02.png",
    "outputs/blockwise/blockwise_f08.png",
    "outputs/blockwise/blockwise_f12.png",
    "outputs/local_affine/benchmark_summary.png",
    "outputs/local_affine/affine_f02.png",
    "outputs/local_affine/affine_f08.png",
    "outputs/local_affine/affine_f12.png",
    "outputs/homography/benchmark_summary.png",
    "outputs/homography/homography_f02.png",
    "outputs/homography/homography_f08.png",
    "outputs/homography/homography_f12.png",
)
DOC_LINKS = {
    "reports/feasibility_report.md": (),
    "reports/blockwise_translation_report.md": (
        "https://colab.research.google.com/drive/1f7KhjyxGqm27Dtg7fNGbjq9gGpetpj3V",
    ),
    "reports/local_affine_report.md": (
        "https://colab.research.google.com/drive/14mlnk92BMVWoRsIjAudiT45A4MhIl1w8",
    ),
    "reports/homography_report.md": (
        "https://colab.research.google.com/drive/1na8VvraDsmXwcJtxwUn9tQRBqmeukroN",
    ),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_mean(
    summary: dict,
    rows: list[dict],
    *,
    mean_key: str,
    row_key: str,
) -> None:
    expected = float(sum(row[row_key] for row in rows) / len(rows))
    actual = float(summary[mean_key])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{mean_key}: stored mean {actual} != {expected}")


def validate_summaries() -> None:
    for folder, prefixes in SUMMARY_PREFIXES.items():
        directory = ROOT / "outputs" / folder
        summary = _read_json(directory / "summary.json")
        rows = summary["results"]
        if summary["flow_indices"] != [2, 8, 12] or len(rows) != 3:
            raise AssertionError(f"{folder}: unexpected DSEC window set")
        with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != len(rows):
            raise AssertionError(f"{folder}: JSON/CSV row count mismatch")
        for prefix in prefixes:
            _assert_mean(
                summary,
                rows,
                mean_key=f"mean_{prefix}_median_epe_px",
                row_key=f"{prefix}_median_endpoint_error_px",
            )
            _assert_mean(
                summary,
                rows,
                mean_key=f"mean_{prefix}_within_2px",
                row_key=f"{prefix}_within_2px",
            )

    baseline = _read_json(ROOT / "outputs" / "summary.json")
    if baseline["flow_indices"] != [2, 8, 12] or len(baseline["results"]) != 9:
        raise AssertionError("baseline: expected nine patch results")


def validate_notebooks() -> None:
    for path in NOTEBOOKS:
        notebook = _read_json(path)
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        if not any('BRANCH = "main"' in source for source in code_cells):
            raise AssertionError(f"{path.name}: setup cell does not pin main")
        for index, source in enumerate(code_cells):
            python_source = "\n".join(
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            compile(python_source, f"{path.name}:cell-{index}", "exec")


def validate_required_artifacts() -> None:
    for relative in REQUIRED_IMAGES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty artifact: {relative}")
    for relative, links in DOC_LINKS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for link in links:
            if link not in text:
                raise AssertionError(f"{relative}: missing {link}")


def validate_checksums() -> None:
    manifest = ROOT / "outputs" / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def main() -> None:
    validate_summaries()
    validate_notebooks()
    validate_required_artifacts()
    validate_checksums()
    print(
        "Repository validation passed: "
        f"{len(NOTEBOOKS)} notebooks, 4 result sets, "
        f"{len(REQUIRED_IMAGES)} required figures."
    )


if __name__ == "__main__":
    main()
