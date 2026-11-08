from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import png


DSEC_SEQUENCE = "thun_00_a"
DSEC_BASE_URL = (
    "https://download.ifi.uzh.ch/rpg/DSEC/train/"
    f"{DSEC_SEQUENCE}/{DSEC_SEQUENCE}"
)
EVENT_URL = f"{DSEC_BASE_URL}_events_left.zip"
FLOW_URL = f"{DSEC_BASE_URL}_optical_flow_forward_event.zip"
FLOW_TIMES_URL = f"{DSEC_BASE_URL}_optical_flow_forward_timestamps.txt"


@dataclass(frozen=True)
class EventWindow:
    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    t_start_us: int
    t_end_us: int

    @property
    def timestamps_ms(self) -> np.ndarray:
        return (self.t_us - self.t_start_us) / 1000.0

    @property
    def normalized_time(self) -> np.ndarray:
        span = float(self.t_end_us - self.t_start_us)
        return (self.t_us.astype(np.float64) - self.t_start_us) / span


def download(
    url: str,
    destination: Path,
    *,
    min_bytes: int = 1024,
    log: Callable[[str], None] = print,
) -> Path:
    """Download a file robustly enough for a Colab runtime."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size >= min_bytes:
        log(f"Cached: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    log(f"Downloading {destination.name} ...")
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=60) as response:
        response_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        next_report = 10
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if response_size:
                    percentage = 100 * downloaded / response_size
                    if percentage >= next_report:
                        log(f"  {percentage:5.1f}%  {downloaded / 1e6:6.1f} MB")
                        next_report += 10
    if downloaded < min_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded only {downloaded} bytes from {url}; expected at least {min_bytes}."
        )
    temporary.replace(destination)
    log(
        f"Downloaded: {destination} "
        f"({downloaded / 1e6:.1f} MB in {time.monotonic() - started:.1f}s)"
    )
    return destination


def prepare_dsec_files(
    data_dir: Path,
    *,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Download the smallest practical DSEC sequence used by this experiment."""
    event_zip = download(
        EVENT_URL,
        data_dir / f"{DSEC_SEQUENCE}_events_left.zip",
        min_bytes=250_000_000,
        log=log,
    )
    flow_zip = download(
        FLOW_URL,
        data_dir / f"{DSEC_SEQUENCE}_optical_flow_forward_event.zip",
        min_bytes=10_000_000,
        log=log,
    )
    flow_times = download(
        FLOW_TIMES_URL,
        data_dir / f"{DSEC_SEQUENCE}_optical_flow_forward_timestamps.txt",
        min_bytes=500,
        log=log,
    )

    event_dir = data_dir / f"{DSEC_SEQUENCE}_left"
    event_dir.mkdir(parents=True, exist_ok=True)
    required = ("events.h5", "rectify_map.h5")
    if not all((event_dir / name).exists() for name in required):
        log("Extracting event archive ...")
        with zipfile.ZipFile(event_zip) as archive:
            archive.extractall(event_dir)

    return {
        "events_h5": event_dir / "events.h5",
        "rectify_h5": event_dir / "rectify_map.h5",
        "flow_zip": flow_zip,
        "flow_times": flow_times,
    }


def load_flow_timestamps(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", delimiter=",", dtype=np.int64)


def load_event_window(
    events_h5: Path,
    t_start_us: int,
    t_end_us: int,
) -> EventWindow:
    """Load DSEC events in a half-open timestamp interval."""
    if t_end_us <= t_start_us:
        raise ValueError("t_end_us must be greater than t_start_us")

    with h5py.File(events_h5, "r") as handle:
        t_offset = int(handle["t_offset"][()])
        relative_start = t_start_us - t_offset
        relative_end = t_end_us - t_offset
        if relative_start < 0:
            raise ValueError("Requested window begins before the sequence.")

        ms_to_idx = np.asarray(handle["ms_to_idx"], dtype=np.int64)
        start_ms = max(0, relative_start // 1000)
        end_ms = min(ms_to_idx.size - 1, (relative_end + 999) // 1000)
        conservative_start = int(ms_to_idx[start_ms])
        conservative_end = int(ms_to_idx[end_ms])

        relative_t = np.asarray(
            handle["events/t"][conservative_start:conservative_end],
            dtype=np.int64,
        )
        x = np.asarray(handle["events/x"][conservative_start:conservative_end])
        y = np.asarray(handle["events/y"][conservative_start:conservative_end])
        polarity = np.asarray(
            handle["events/p"][conservative_start:conservative_end]
        )

    keep = (relative_t >= relative_start) & (relative_t < relative_end)
    return EventWindow(
        x=x[keep],
        y=y[keep],
        t_us=relative_t[keep] + t_offset,
        polarity=polarity[keep],
        t_start_us=t_start_us,
        t_end_us=t_end_us,
    )


def load_rectification_map(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["rectify_map"][()])


def rectify_xy(
    x: np.ndarray,
    y: np.ndarray,
    rectify_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width, channels = rectify_map.shape
    if channels != 2:
        raise ValueError(f"Expected a 2-channel rectification map, got {channels}.")
    if np.any(x >= width) or np.any(y >= height) or np.any(x < 0) or np.any(y < 0):
        raise ValueError("Raw event coordinates fall outside the rectification map.")
    mapped = rectify_map[y.astype(np.int32), x.astype(np.int32)]
    return mapped[:, 0], mapped[:, 1]


def decode_flow_png(data: bytes | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode DSEC's 3-channel uint16 flow PNG without 8-bit truncation."""
    source = io.BytesIO(data) if isinstance(data, bytes) else str(data)
    width, height, rows, metadata = png.Reader(source).read()
    if metadata.get("bitdepth") != 16 or metadata.get("planes") != 3:
        raise ValueError(f"Unexpected flow PNG metadata: {metadata}")
    raw = np.vstack([np.asarray(row, dtype=np.uint16) for row in rows])
    raw = raw.reshape(height, width, 3)
    flow_xy = (raw[:, :, :2].astype(np.float32) - 32768.0) / 128.0
    valid = raw[:, :, 2] == 1
    return flow_xy, valid


def flow_png_name(flow_index: int) -> str:
    return f"{2 * flow_index + 2:06d}.png"


def load_flow_target(
    flow_zip: Path,
    flow_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(flow_zip) as archive:
        return decode_flow_png(archive.read(flow_png_name(flow_index)))
