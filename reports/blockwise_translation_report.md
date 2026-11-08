# Block-wise Translation Experiment

Status: validated historical stage. This model remains B, the stable fallback
for BA and BH. The current default is local affine (BA).

## Scope

This experiment replaces one global translation with overlapping local
translations while retaining the event-only contrast objective.

No affine or homography model is used.

## Configuration

- DSEC sequence: `thun_00_a`, left event camera;
- windows: forward-flow indices `2`, `8`, and `12`;
- window duration: approximately 100 ms;
- event volume: 9 temporal bins with signed polarity;
- tile size: `64 x 64`;
- tile stride: `32`;
- phase-correlation initialization: centered single-bin short baselines plus
  first/last 3-bin long baseline;
- contrast refinement: +/-3 px around the phase-correlation vector;
- block displacement bound: +/-48 px;
- output field: bilinear interpolation through overlapping tile centers;
- low-support tiles: global-translation fallback and nearest valid neighbor.

## Result

The primary evaluation compares predicted endpoint displacement with DSEC
forward flow at every valid labeled pixel. The previous baseline evaluation
used a median flow vector per patch, so the new dense per-pixel EPE is stricter
and should not be compared directly with the old 1.35 px patch result.

| window | global median EPE | block-wise median EPE | global <= 2 px | block-wise <= 2 px | warp coverage |
|---:|---:|---:|---:|---:|---:|
| 2 | 20.51 px | 1.61 px | 1.0% | 58.1% | 98.9% |
| 8 | 23.58 px | 2.03 px | 0.8% | 49.6% | 98.9% |
| 12 | 11.73 px | 1.64 px | 0.8% | 59.0% | 98.8% |
| mean | 18.61 px | 1.76 px | 0.8% | 55.6% | 98.8% |

The mean median-EPE reduction is approximately 90.5%. The full 640 x 480 event
image is therefore working as a first block-wise baseline.

After inputs are cached, all three windows take about 40 seconds on the local
ARM CPU, or about 13 seconds per 100 ms window. The implementation is a Python
reference implementation, not yet an optimized real-time path.

On a free Colab x86 CPU runtime, the same three windows take approximately
190.9 seconds after data preparation, or 59.7-62.4 seconds per 100 ms window.
The complete first Colab run, including the 299 MB event download, took about
3 minutes 37 seconds and remained within the 12.67 GB runtime memory limit.

Final executed notebook:
[DSEC_DVS_Blockwise_Translation_PhaseFix_FINAL](https://colab.research.google.com/drive/1f7KhjyxGqm27Dtg7fNGbjq9gGpetpj3V).

## Visual result

The block-wise compensated image preserves substantially more structure than a
single global translation. Remaining curved and repeated ghost contours are
expected because each tile still assumes pure translation and the 100 ms window
contains rotation and depth change.

![Block-wise comparison](../outputs/blockwise/benchmark_summary.png)

Qualitative results:
[window 2](../outputs/blockwise/blockwise_f02.png),
[window 8](../outputs/blockwise/blockwise_f08.png),
[window 12](../outputs/blockwise/blockwise_f12.png).

## Initialization failure and correction

The first implementation used phase correlation over the first 3 and last 3
temporal bins, then refined only +/-3 px around that result. At flow index 8,
the tile covering `(y=240,x=96)` had:

- DSEC median flow: `(-17.63, +3.71) px`;
- long-baseline phase result: `(0, 0)` with a misleadingly high peak score;
- early-to-middle single-bin phase result: `(-18.05, +4.54) px`;
- middle-to-late single-bin phase result: `(-17.54, +4.25) px`;
- final pre-fix estimate: `(-3, +3) px`.

The zero peak was not rejected because short-baseline evidence was never
computed. The fixed initialization uses centered single bins at normalized
time approximately 0, 0.5, and 1.0. When both short-baseline vectors agree
within `max(6 px, 30%)`, their average is preferred; otherwise the estimate
falls back to the long temporal baseline or the global motion. Refinement
cropping now accounts for the actual per-slice temporal warp factor rather
than treating the full endpoint displacement as a single-slice shift.

### Before/after

| case | before | after |
|---|---:|---:|
| all windows, mean median EPE | 2.37 px | **1.76 px** |
| all windows, mean EPE <= 2 px | 45.1% | **55.6%** |
| failure patch, window 2 | 12.22 px | **1.39 px** |
| failure patch, window 8 | 15.78 px | **2.65 px** |
| failure patch, window 12 | 14.99 px | **1.76 px** |

This confirms an avoidable estimation-policy defect. The slightly lower warp
coverage (`99.2%` to `98.8%`) results from retaining larger, more accurate
local displacements near image boundaries.

## Residual model limitation

The corrected lower-left failure patch still has 1.39-2.65 px per-pixel median
EPE. Its DSEC flow varies within the patch because of near-field parallax and
non-linear motion, while block-wise translation represents only one vector per
interpolated region. This residual belongs to the model boundary and is not
expected to disappear without local affine, homography, or motion segmentation.

The full-frame result is improved, but block-wise translation is still not a
complete solution. Local affine was subsequently validated as BA and is the
current default. Local homography was evaluated independently as BH and is
retained as a neutral optional result.

## Acceptance evidence

- synthetic phase-correlation test passes;
- synthetic spurious-zero long-baseline regression test passes;
- synthetic two-region translation test passes;
- existing constant-translation test passes;
- complete DSEC replay executes without error;
- warp coverage is at least 98.7% in all three windows;
- block-wise median EPE is below 2.5 px in all three evaluation windows;
- all result data and representative images are generated from one command.

## Reproduce

```bash
python scripts/run_blockwise_demo.py \
  --data-dir data \
  --output-dir outputs/blockwise \
  --flow-indices 2 8 12 \
  --tile-size 64 \
  --stride 32
```
