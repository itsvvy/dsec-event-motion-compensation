# DVS Event-Only Motion Compensation on DSEC

CPU-first feasibility validation for the event preprocessing stage in the
proposal **"Edge-Based Dual Stereo Sensing System: Fusion of CIS and DVS
Cameras for Real-Time Obstacle Detection and Depth Estimation."**

This repository tests **motion compensation of the DVS event image itself**.
It does not deblur a CIS RGB image, perform image-to-event reconstruction, or
estimate dense optical flow with a neural network.

## Algorithm

For a finite event window `[t0, t1]`, events are split into temporal bins and
accumulated with sign by polarity. A constant local displacement
`theta = (vx, vy)` is searched coarse-to-fine. A slice at normalized time
`tau` is warped toward the window midpoint:

```text
x' = x - vx * (tau - 0.5)
y' = y - vy * (tau - 0.5)
```

The objective is the variance of the polarity-weighted, lightly smoothed
event image after all slices are accumulated. The estimate is the displacement
over the complete window, not velocity per millisecond.

The implementation is intentionally small and deterministic:

- no training and no GPU requirement;
- DSEC events, timestamps, and calibration are read directly;
- event windows are selected from the official forward-flow timeline;
- DSEC's 16-bit forward optical flow provides an independent correctness check
  in each patch;
- the full run covers three 100 ms windows and three spatial patches on CPU.

## DSEC sample

The experiment uses the smallest practical driving sequence:

- sequence: `thun_00_a`, left event camera;
- event archive: about 285 MB;
- forward optical-flow labels: about 11 MB;
- event-camera resolution: 640 x 480.

Files are downloaded at runtime from the official
[DSEC download page](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
Dataset license: CC BY-SA 4.0.

## Reproduce locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'from tests.test_synthetic import test_recovers_known_constant_translation as t; t()'
python scripts/run_dsec_demo.py
```

The first run downloads and extracts the event sequence. Subsequent runs reuse
`data/`. Results are written to `outputs/`:

- `summary.json`: all requested metrics;
- `results.csv`: flat experiment table;
- `dsec_fXX_yYYY_xXXX.png`: before/after images and objective surface.

## Colab

Open `notebooks/dsec_event_motion_compensation.ipynb` in Google Colab. It uses
the free CPU runtime, installs only the small scientific Python dependencies,
clones this repository, runs the test and benchmark, and displays the results.

## Reference results

Local CPU result for three 100 ms windows and three 128 x 128 patches:

| metric | result |
|---|---:|
| median endpoint error vs. DSEC flow | 1.35 px |
| mean endpoint error | 3.37 px |
| median event-image variance gain | 3.66x |
| mean gradient-energy gain | 8.51x |
| benchmark runtime after inputs are cached | 4.63 s |

The endpoint error is measured against the median valid optical-flow vector in
each patch. This is an approximation because DSEC's 100 ms flow field contains
rotation, parallax, and independently moving objects rather than one exact
global translation.

Two patches track DSEC flow consistently. A textured left-side patch is a
deliberate failure case: contrast still increases, but the estimated motion
does not match the patch median because multiple motions coexist. This shows
why a single full-frame translation is insufficient for a moving driving scene
and why the production design should use local/multi-model compensation.

## Scope and limitations

- The experiment validates the algorithm and CPU data path, not RK3588
  deployment latency.
- The 100 ms DSEC windows are intentionally more aggressive than the likely
  10-50 ms windows on the eventual hardware.
- A constant-translation patch model cannot represent rotation, depth
  parallax, or independently moving objects.
- The contrast objective can select a coherent sub-population rather than the
  patch median; flow-supervised evaluation is therefore necessary.
- A practical next step is block-wise motion estimation plus homography or
  locally fitted affine motion, followed by validation on 20-50 ms windows.

## Citations

If this experiment is used academically, cite:

```bibtex
@article{Gehrig21ral,
  author  = {Mathias Gehrig and Willem Aarents and Daniel Gehrig and Davide Scaramuzza},
  title   = {DSEC: A Stereo Event Camera Dataset for Driving Scenarios},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2021},
  doi     = {10.1109/LRA.2021.3068942}
}
```
