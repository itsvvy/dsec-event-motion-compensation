# Local Homography Experiment

## Scope and independence

This experiment compares three models on identical DSEC event windows:

- **B**: the validated block-wise translation baseline;
- **BA**: block-wise fallback plus a directly estimated local affine field;
- **BH**: block-wise fallback plus a directly estimated local homography field.

BH is independent from BA. It does not read, initialize from, or blend the BA
field. B provides the initialization and per-region fallback used by both BA
and BH.

No neural network, CIS deblurring, or RK3588 implementation is introduced.

## Method

- B uses a `64 x 64` tile with 32 px stride.
- BA and BH use overlapping `160 x 160` regions with 80 px stride.
- BH estimates an 8-DoF homography from the first/last three temporal event
  bins with OpenCV ECC.
- The local early-to-late homography is converted to a per-pixel endpoint
  displacement field through the perspective projection. Accepted regions
  are tapered and blended into the B dense field.
- A homography region is accepted only when:
  - ECC correlation is at least `0.30`;
  - the center endpoint displacement is at most `48 px`;
  - motion variation inside the region is at most `32 px`;
  - the perspective denominator remains at least `0.10`;
  - event-image variance is at least `1.02x` the B result.
- Rejected regions retain the B field exactly.

The same DSEC sequence (`thun_00_a`, left event camera), 100 ms windows
`2/8/12`, nine temporal bins, flow evaluation, and target patches are used for
all three models.

## Accuracy comparison

| window | B median EPE | BA median EPE | BH median EPE | B <= 2 px | BA <= 2 px | BH <= 2 px | BH accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.61 px | 1.47 px | **1.44 px** | 58.1% | 65.2% | **67.0%** | 3/35 |
| 8 | 2.03 px | 1.61 px | **1.53 px** | 49.6% | 58.9% | **59.3%** | 8/35 |
| 12 | 1.64 px | **1.47 px** | 1.58 px | 59.0% | **67.1%** | 62.0% | 6/35 |
| mean | 1.76 px | 1.519 px | **1.517 px** | 55.6% | **63.7%** | 62.8% | - |

The mean median-EPE difference between BA and BH is only about `0.002 px`.
BH has lower mean endpoint error in window 2 but higher mean error in windows 8
and 12. It also has more conservative acceptance than BA: 17 accepted regions
overall versus 22 for BA.

| window | B mean EPE | BA mean EPE | BH mean EPE |
|---:|---:|---:|---:|
| 2 | 4.38 px | **3.05 px** | 3.10 px |
| 8 | 5.69 px | **3.78 px** | 3.85 px |
| 12 | 3.72 px | **2.68 px** | 2.83 px |

The low median EPE but worse mean EPE shows that perspective freedom can improve
many confident pixels while creating larger outliers in weak regions.

## Target patch

At `(y=240, x=96)`:

| window | B | BA | BH |
|---:|---:|---:|---:|
| 2 | 1.39 px | 1.39 px | 1.39 px |
| 8 | 2.65 px | **1.94 px** | 1.96 px |
| 12 | 1.76 px | 1.76 px | 1.76 px |

Homography does not improve the previously difficult patch over affine. Windows
2 and 12 retain the B result in all models because the local models are not
accepted there.

## Additional metrics

| metric | B | BA | BH |
|---|---:|---:|---:|
| mean median EPE | 1.760 px | 1.519 px | **1.517 px** |
| mean EPE <= 2 px | 55.6% | **63.7%** | 62.8% |
| mean variance gain | 3.668x | **3.741x** | 3.709x |
| mean warp coverage | 98.85% | 98.83% | 98.85% |

## Runtime

On the free Colab x86 CPU runtime:

- full first run including dataset download: about 2 minutes 42 seconds;
- cached three-window benchmark: 133.6 seconds;
- per-window total: 42.2-43.8 seconds;
- homography estimation itself: 1.74-1.94 seconds per window;
- observed RAM use remained below 2.5 GB out of 12.67 GB.

The shared B stage dominates runtime; homography adds little incremental cost
after the block field exists.

## Sensitivity check

A more conservative gate (`correlation >= 0.50`, contrast `>= 1.04`,
variation `<= 20 px`) rejects all regions on window 2 and produces mean BH
median EPE `1.642 px`. It is worse than both the original BH configuration and
BA, so it is not used.

## Conclusion

Local homography is technically valid and occasionally provides the best
window-level median EPE, but it does not provide a robust improvement over BA
on this driving sample. The extra two perspective parameters are not reliably
supported by non-planar depth variation, near-field parallax, and multiple
motion layers. BA remains the practical default; BH should be retained as a
selected-region model for future data where planar support can be identified
more reliably.

## Reproduce

```bash
python scripts/run_homography_demo.py \
  --data-dir data \
  --output-dir outputs/homography \
  --flow-indices 2 8 12 \
  --tile-size 64 \
  --stride 32
```

## Acceptance evidence

- all baseline, block-wise, affine, and homography synthetic tests pass;
- rejected homography regions preserve the B field exactly;
- BH is computed independently from BA;
- complete DSEC replay covers windows 2, 8, and 12;
- raw JSON, CSV, and benchmark figure are retained;
- no branch history or published tags were rewritten.
