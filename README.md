# lss-sbi-lab

A compact research prototype for **calibrated simulation-based inference from a 3-D galaxy field**. The point is not to claim precision cosmology from a toy simulator. The point is to expose and measure a central failure mode: an amortised posterior can become biased and overconfident when the real galaxy–matter relation differs from the training forward model.

## Scientific question

How reliable is neural posterior estimation for \((\Omega_m,\sigma_8,b_1)\) when the training simulations assume linear galaxy bias but the observation contains an unmodelled nonlinear-bias contribution?

## Forward model

1. Draw \((\Omega_m,\sigma_8,b_1)\) from bounded uniform priors.
2. Generate a 3-D Gaussian matter field with a BBKS-shaped linear power spectrum.
3. Normalise the **discrete, grid-resolved** field so that the expected 8 Mpc/h top-hat RMS is \(\sigma_8\).
4. Apply linear Kaiser redshift-space distortions through \(b_1+f\mu^2\), with \(f\simeq\Omega_m^{0.55}\).
5. Convert the field to a positive lognormal galaxy intensity.
6. Poisson-sample galaxy counts, creating actual tracer shot noise.
7. Measure the monopole \(P_0(k)\), quadrupole \(P_2(k)\), and four count-field non-Gaussian summaries.

## Inference and validation

The script trains a conditional diagonal Gaussian-mixture density estimator for \(q(\theta\mid x)\). Parameters are represented in a prior-bounded logit space, so posterior samples cannot leave the stated prior support.

Outputs include:

- a held-out posterior corner plot;
- simulation-based-calibration rank histograms;
- empirical coverage curves with a finite-sample band;
- a nonlinear-bias misspecification stress test;
- feature-space out-of-distribution warning scores;
- machine-readable metrics and saved posterior samples.

## Run

```bash
python -m pip install -r requirements.txt
python sbi_lss_lab.py --quick --output-dir results/quick
```

A tiny integration run is available for software checks:

```bash
python sbi_lss_lab.py --smoke --output-dir results/smoke
```

A larger run:

```bash
python sbi_lss_lab.py \
  --n-train 3000 \
  --n-validation 400 \
  --n-calibration 150 \
  --epochs 300 \
  --output-dir results/full
```

Run the unit checks with:

```bash
python -m unittest discover -s tests -v
```

## Main outputs

- `posterior_corner.png`
- `calibration.png`
- `misspecification.png`
- `training_history.png`
- `summary_example.png`
- `metrics.json`
- `diagnostics.npz`
- `posterior_model.pt`

## Deliberate limitations

This is a controlled methods sandbox, not a survey analysis.

- BBKS omits baryon acoustic features and should be replaced by CAMB/CLASS or simulation products.
- The matter field is Gaussian rather than evolved with an N-body or particle-mesh solver.
- The biased lognormal Cox process is a transparent tracer model, not a complete HOD/SHAM prescription.
- The box is periodic and has no angular mask, radial selection, fibre assignment, redshift errors, or window-function convolution.
- The summary vector is not field-level inference.
- The neural density estimator is intentionally lightweight; production work should compare multiple posterior estimators and simulation budgets.

## High-impact next steps

1. Replace the field generator with Quijote/CAMELS or a differentiable particle-mesh forward model.
2. Add survey selection and a random-catalogue-aware estimator.
3. Compare power-spectrum-only, higher-order-summary, and learned field embeddings.
4. Train over nonlinear-bias and HOD nuisance families rather than merely stress-testing them.
5. Add TARP/local-coverage checks and simulation-based model criticism.
6. Extend the parameter space to primordial non-Gaussianity using a physically validated simulator.

`legacy_sbi_lss_lab.py` preserves the original starter script for comparison.

## Nuisance-marginalised nonlinear-bias experiment

After producing a baseline directory with `sbi_lss_lab.py`, train a second
posterior that marginalises over an unobserved quadratic-bias coefficient:

```bash
python robust_bias_lab.py \
  --baseline-dir results/baseline_final \
  --output-dir results/robust_final \
  --b2-low -0.20 \
  --b2-high 0.20 \
  --stress-b2 0.15 \
  --n-train 5000 \
  --n-validation 750 \
  --n-calibration 500 \
  --n-comparison 200 \
  --n-posterior-samples 2000 \
  --epochs 250
```

The robust network still predicts only `(Omega_m, sigma_8, b1)`.  The latent
`b2` values are drawn during simulation and integrated out by amortised
training.  The script loads the existing baseline model rather than retraining
it, evaluates both posteriors on identical paired universes, and writes all
per-realisation truths, medians, intervals, coverage flags, OOD scores, and
random seeds to `robust_diagnostics.npz`.

Key outputs:

- `robust_metrics.json`: aggregate calibration and baseline-versus-robust results;
- `robust_diagnostics.npz`: all per-realisation values for bootstrap/statistical checks;
- `baseline_vs_robust.png`: coverage, RMSE, and interval-width comparison;
- `robust_calibration.png`: calibration under the nuisance prior;
- `robust_posterior_model.pt`: trained nuisance-marginalised posterior.

A smoke test is available with:

```bash
python robust_bias_lab.py \
  --baseline-dir results/reference_run \
  --output-dir results/robust_smoke \
  --smoke
```
