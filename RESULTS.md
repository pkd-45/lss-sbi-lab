# Reference-run results

These numbers come from the deterministic default command:

```bash
python sbi_lss_lab.py --output-dir results/reference_run
```

Configuration: 2,000 training simulations, 300 validation simulations, 100 calibration simulations, 30 paired misspecification tests, a 32³ grid in a 500 Mpc/h periodic box, and 1,000 posterior samples per calibration case.

## Well-specified calibration

At a nominal 68% credibility level, empirical coverage was:

| Parameter | Empirical coverage |
|---|---:|
| Omega_m | 0.69 |
| sigma_8 | 0.68 |
| b_1 | 0.69 |

Coverage-curve RMSE relative to the nominal diagonal was 0.038, 0.023, and 0.038, respectively. This indicates that the lightweight neural posterior is adequately calibrated for the simulator family used for training.

## Paired nonlinear-bias mismatch

The stress test adds a quadratic tracer-bias contribution absent from training while keeping the cosmological parameters and random initial phases paired between the baseline and mismatch simulations.

| Parameter | 68% coverage: trained model | 68% coverage: mismatch | Posterior-median RMSE ratio |
|---|---:|---:|---:|
| Omega_m | 0.63 | 0.47 | 2.73 |
| sigma_8 | 0.67 | 0.33 | 1.58 |
| b_1 | 0.73 | 0.07 | 3.24 |

The median summary-space Mahalanobis RMS warning score rises from 0.75 for well-specified mocks to 1.73 for misspecified mocks. The warning score is not a calibrated p-value, but it correctly identifies that the mismatched observations lie farther from the training-simulator support.

## Interpretation

The central result is not that this toy simulator measures cosmology accurately. It is that an amortised posterior can pass standard coverage tests on in-distribution simulations yet fail severely under a plausible tracer-model mismatch. The immediate research implication is to train over flexible nuisance families, perform local coverage/model-criticism checks, and test robustness across independent simulation suites before applying field-level or learned-summary inference to survey data.

## Limitations

All conclusions are conditional on the controlled simulator described in the README. No claim is made about DESI, Euclid, Rubin, primordial non-Gaussianity, or precision survey constraints. The next scientific upgrade is to repeat the same calibration and mismatch logic using Quijote/CAMELS or a particle-mesh forward model with explicit survey selection.
