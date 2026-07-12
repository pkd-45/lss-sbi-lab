#!/usr/bin/env python3
"""Nuisance-marginalised nonlinear-bias experiment for ``sbi_lss_lab``.

This script trains a second amortised posterior on simulations whose quadratic
bias coefficient b2 is drawn from a user-defined prior, while b2 is *not* an
inferred output.  The network therefore learns q(Omega_m, sigma_8, b1 | x)
with b2 marginalised by simulation.

It then loads an existing baseline posterior trained at b2=0 and compares both
models on exactly the same paired universes at b2=0 and at a fixed stress-test
value (default b2=0.15).

The baseline is not retrained.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sbi_lss_lab import (
    CREDIBILITY_LEVELS,
    PARAMETER_LABELS,
    PARAMETER_NAMES,
    PRIOR_HIGH,
    PRIOR_LOW,
    BoundedParameterTransform,
    GalaxyFieldSimulator,
    MixturePosteriorNetwork,
    RunConfig,
    SimulationConfig,
    Standardizer,
    TrainedPosterior,
    central_interval,
    plot_calibration,
    plot_training_history,
    resolve_device,
    sample_prior,
    set_global_seed,
    summary_ood_diagnostics,
    train_posterior,
)

Array = np.ndarray


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint across recent PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_posterior(checkpoint_path: Path, device: torch.device) -> tuple[TrainedPosterior, SimulationConfig, RunConfig]:
    checkpoint = _torch_load(checkpoint_path, device)
    sim_config = SimulationConfig(**checkpoint["simulation_config"])
    run_config = RunConfig(**checkpoint["run_config"])
    simulator = GalaxyFieldSimulator(sim_config)

    model = MixturePosteriorNetwork(
        input_features=simulator.summary_size,
        parameter_features=len(PARAMETER_NAMES),
        hidden_features=run_config.hidden_features,
        n_components=run_config.n_components,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x_scaler = Standardizer(
        mean=np.asarray(checkpoint["x_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["x_std"], dtype=np.float32),
    )
    theta_latent_scaler = Standardizer(
        mean=np.asarray(checkpoint["theta_latent_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["theta_latent_std"], dtype=np.float32),
    )
    theta_scaler = BoundedParameterTransform(
        low=np.asarray(checkpoint["prior_low"], dtype=np.float32),
        high=np.asarray(checkpoint["prior_high"], dtype=np.float32),
        latent_scaler=theta_latent_scaler,
    )
    return (
        TrainedPosterior(
            model=model,
            x_scaler=x_scaler,
            theta_scaler=theta_scaler,
            device=device,
        ),
        sim_config,
        run_config,
    )


def build_marginalised_dataset(
    simulator: GalaxyFieldSimulator,
    n_samples: int,
    seed: int,
    b2_low: float,
    b2_high: float,
    label: str,
) -> tuple[Array, Array, Array]:
    """Generate theta, summaries, and latent nuisance values."""

    if b2_low > b2_high:
        raise ValueError("b2_low must not exceed b2_high")
    rng = np.random.default_rng(seed)
    theta = sample_prior(rng, n_samples)
    b2 = rng.uniform(b2_low, b2_high, size=n_samples).astype(np.float32)
    summaries = np.empty((n_samples, simulator.summary_size), dtype=np.float32)
    progress_step = max(n_samples // 10, 1)
    for index, parameters in enumerate(theta):
        summaries[index] = simulator.simulate(
            parameters,
            seed=seed + 100_000 + index,
            nonlinear_bias_b2=float(b2[index]),
        )
        if (index + 1) % progress_step == 0 or index + 1 == n_samples:
            print(f"  {label}: simulated {index + 1}/{n_samples}")
    return theta, summaries, b2


def run_marginal_calibration(
    posterior: TrainedPosterior,
    simulator: GalaxyFieldSimulator,
    n_tests: int,
    n_posterior_samples: int,
    b2_low: float,
    b2_high: float,
    seed: int,
) -> dict[str, Array]:
    """SBC/coverage under the same nuisance prior used for training."""

    rng = np.random.default_rng(seed)
    truths = sample_prior(rng, n_tests)
    nuisance_b2 = rng.uniform(b2_low, b2_high, size=n_tests).astype(np.float32)
    ranks = np.zeros((n_tests, len(PARAMETER_NAMES)), dtype=np.int64)
    covered = np.zeros(
        (n_tests, len(PARAMETER_NAMES), len(CREDIBILITY_LEVELS)), dtype=bool
    )
    medians = np.zeros_like(truths)
    lower68 = np.zeros_like(truths)
    upper68 = np.zeros_like(truths)

    for index, truth in enumerate(truths):
        summary = simulator.simulate(
            truth,
            seed=seed + 200_000 + index,
            nonlinear_bias_b2=float(nuisance_b2[index]),
        )
        samples = posterior.sample(
            summary,
            n_posterior_samples,
            np.random.default_rng(seed + 300_000 + index),
        )
        ranks[index] = np.sum(samples < truth[None, :], axis=0)
        medians[index] = np.median(samples, axis=0)
        lower68[index], upper68[index] = central_interval(samples, 0.68)
        for level_index, level in enumerate(CREDIBILITY_LEVELS):
            lower, upper = central_interval(samples, float(level))
            covered[index, :, level_index] = (truth >= lower) & (truth <= upper)
        if (index + 1) % max(n_tests // 5, 1) == 0 or index + 1 == n_tests:
            print(f"  robust calibration: completed {index + 1}/{n_tests}")

    return {
        "truths": truths,
        "b2": nuisance_b2,
        "ranks": ranks,
        "coverage": np.mean(covered, axis=0),
        "covered": covered,
        "medians": medians,
        "lower68": lower68,
        "upper68": upper68,
    }


def evaluate_two_posteriors(
    baseline: TrainedPosterior,
    robust: TrainedPosterior,
    simulator: GalaxyFieldSimulator,
    baseline_training_summaries: Array,
    robust_training_summaries: Array,
    n_tests: int,
    n_posterior_samples: int,
    stress_b2: float,
    seed: int,
) -> dict[str, Any]:
    """Evaluate baseline and robust models on shared paired universes."""

    rng = np.random.default_rng(seed)
    truths = sample_prior(rng, n_tests)
    seeds = seed + 500_000 + np.arange(n_tests, dtype=np.int64)
    model_names = ("baseline", "robust")
    condition_names = ("b2_zero", "b2_stress")

    arrays: dict[str, Array] = {"truths": truths, "simulation_seeds": seeds}
    for model_name in model_names:
        for condition in condition_names:
            prefix = f"{model_name}_{condition}"
            arrays[f"{prefix}_median"] = np.zeros_like(truths, dtype=np.float32)
            arrays[f"{prefix}_lower68"] = np.zeros_like(truths, dtype=np.float32)
            arrays[f"{prefix}_upper68"] = np.zeros_like(truths, dtype=np.float32)
            arrays[f"{prefix}_covered68"] = np.zeros_like(truths, dtype=bool)
            arrays[f"{prefix}_ood"] = np.zeros(n_tests, dtype=np.float32)

    for index, truth in enumerate(truths):
        summary_zero = simulator.simulate(truth, int(seeds[index]), nonlinear_bias_b2=0.0)
        summary_stress = simulator.simulate(
            truth, int(seeds[index]), nonlinear_bias_b2=stress_b2
        )
        for model_index, (model_name, posterior, training_x) in enumerate(
            (
                ("baseline", baseline, baseline_training_summaries),
                ("robust", robust, robust_training_summaries),
            )
        ):
            for condition_index, (condition, summary) in enumerate(
                (("b2_zero", summary_zero), ("b2_stress", summary_stress))
            ):
                samples = posterior.sample(
                    summary,
                    n_posterior_samples,
                    np.random.default_rng(
                        seed + 600_000 + 10_000 * model_index + 1_000 * condition_index + index
                    ),
                )
                prefix = f"{model_name}_{condition}"
                median = np.median(samples, axis=0)
                lower, upper = central_interval(samples, 0.68)
                arrays[f"{prefix}_median"][index] = median
                arrays[f"{prefix}_lower68"][index] = lower
                arrays[f"{prefix}_upper68"][index] = upper
                arrays[f"{prefix}_covered68"][index] = (
                    (truth >= lower) & (truth <= upper)
                )
                arrays[f"{prefix}_ood"][index] = summary_ood_diagnostics(
                    summary, training_x, posterior.x_scaler
                )["mahalanobis_rms"]

        if (index + 1) % max(n_tests // 5, 1) == 0 or index + 1 == n_tests:
            print(f"  baseline-vs-robust: completed {index + 1}/{n_tests}")

    metrics: dict[str, Any] = {
        "n_tests": int(n_tests),
        "stress_b2": float(stress_b2),
        "models": {},
    }
    for model_name in model_names:
        metrics["models"][model_name] = {}
        for condition in condition_names:
            prefix = f"{model_name}_{condition}"
            error = arrays[f"{prefix}_median"] - truths
            width = arrays[f"{prefix}_upper68"] - arrays[f"{prefix}_lower68"]
            covered = arrays[f"{prefix}_covered68"]
            model_condition: dict[str, Any] = {
                "ood_median": float(np.median(arrays[f"{prefix}_ood"])),
                "ood_90th_percentile": float(np.quantile(arrays[f"{prefix}_ood"], 0.9)),
                "parameters": {},
            }
            for parameter_index, name in enumerate(PARAMETER_NAMES):
                coverage = float(np.mean(covered[:, parameter_index]))
                standard_error = math.sqrt(max(coverage * (1.0 - coverage), 0.0) / n_tests)
                model_condition["parameters"][name] = {
                    "coverage_68": coverage,
                    "coverage_68_wald95": [
                        max(0.0, coverage - 1.96 * standard_error),
                        min(1.0, coverage + 1.96 * standard_error),
                    ],
                    "median_bias": float(np.median(error[:, parameter_index])),
                    "median_absolute_error": float(
                        np.median(np.abs(error[:, parameter_index]))
                    ),
                    "posterior_median_rmse": float(
                        np.sqrt(np.mean(error[:, parameter_index] ** 2))
                    ),
                    "median_68_width": float(np.median(width[:, parameter_index])),
                }
            metrics["models"][model_name][condition] = model_condition

    metrics["stress_mitigation"] = {}
    for parameter_index, name in enumerate(PARAMETER_NAMES):
        baseline_metrics = metrics["models"]["baseline"]["b2_stress"]["parameters"][name]
        robust_metrics = metrics["models"]["robust"]["b2_stress"]["parameters"][name]
        metrics["stress_mitigation"][name] = {
            "coverage_gain": robust_metrics["coverage_68"] - baseline_metrics["coverage_68"],
            "rmse_ratio_robust_over_baseline": (
                robust_metrics["posterior_median_rmse"]
                / max(baseline_metrics["posterior_median_rmse"], 1.0e-12)
            ),
            "width_ratio_robust_over_baseline": (
                robust_metrics["median_68_width"]
                / max(baseline_metrics["median_68_width"], 1.0e-12)
            ),
            "median_paired_absolute_error_improvement": float(
                np.median(
                    np.abs(arrays["baseline_b2_stress_median"][:, parameter_index] - truths[:, parameter_index])
                    - np.abs(arrays["robust_b2_stress_median"][:, parameter_index] - truths[:, parameter_index])
                )
            ),
        }

    return {"metrics": metrics, "arrays": arrays}


def plot_model_comparison(metrics: dict[str, Any], path: Path) -> None:
    x = np.arange(len(PARAMETER_NAMES), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    baseline_stress = metrics["models"]["baseline"]["b2_stress"]["parameters"]
    robust_stress = metrics["models"]["robust"]["b2_stress"]["parameters"]

    baseline_cov = [baseline_stress[name]["coverage_68"] for name in PARAMETER_NAMES]
    robust_cov = [robust_stress[name]["coverage_68"] for name in PARAMETER_NAMES]
    axes[0].bar(x - width / 2, baseline_cov, width, label="baseline")
    axes[0].bar(x + width / 2, robust_cov, width, label="marginalised")
    axes[0].axhline(0.68, linestyle="--", label="nominal 68%")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("empirical 68% coverage")
    axes[0].set_xticks(x, PARAMETER_LABELS)
    axes[0].legend(fontsize=8)

    baseline_rmse = [baseline_stress[name]["posterior_median_rmse"] for name in PARAMETER_NAMES]
    robust_rmse = [robust_stress[name]["posterior_median_rmse"] for name in PARAMETER_NAMES]
    axes[1].bar(x - width / 2, baseline_rmse, width, label="baseline")
    axes[1].bar(x + width / 2, robust_rmse, width, label="marginalised")
    axes[1].set_ylabel("posterior-median RMSE")
    axes[1].set_xticks(x, PARAMETER_LABELS)

    baseline_width = [baseline_stress[name]["median_68_width"] for name in PARAMETER_NAMES]
    robust_width = [robust_stress[name]["median_68_width"] for name in PARAMETER_NAMES]
    axes[2].bar(x - width / 2, baseline_width, width, label="baseline")
    axes[2].bar(x + width / 2, robust_width, width, label="marginalised")
    axes[2].set_ylabel("median 68% interval width")
    axes[2].set_xticks(x, PARAMETER_LABELS)

    fig.suptitle(f"Fixed nonlinear-bias stress test: $b_2={metrics['stress_b2']:.2f}$")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_robust_checkpoint(
    posterior: TrainedPosterior,
    sim_config: SimulationConfig,
    run_config: RunConfig,
    b2_low: float,
    b2_high: float,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": posterior.model.state_dict(),
            "x_mean": posterior.x_scaler.mean,
            "x_std": posterior.x_scaler.std,
            "theta_latent_mean": posterior.theta_scaler.latent_scaler.mean,
            "theta_latent_std": posterior.theta_scaler.latent_scaler.std,
            "prior_low": PRIOR_LOW,
            "prior_high": PRIOR_HIGH,
            "simulation_config": asdict(sim_config),
            "run_config": asdict(run_config),
            "training_nuisance_prior": {
                "nonlinear_bias_b2_low": float(b2_low),
                "nonlinear_bias_b2_high": float(b2_high),
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test a nonlinear-bias-marginalised cosmological posterior."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("results/baseline_final"),
        help="Directory containing baseline posterior_model.pt and training_dataset.npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/robust_final"))
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--b2-low", type=float, default=-0.20)
    parser.add_argument("--b2-high", type=float, default=0.20)
    parser.add_argument("--stress-b2", type=float, default=0.15)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-validation", type=int, default=750)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--n-comparison", type=int, default=200)
    parser.add_argument("--n-posterior-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_train = 80
        args.n_validation = 20
        args.n_calibration = 5
        args.n_comparison = 4
        args.n_posterior_samples = 80
        args.epochs = 5
        args.batch_size = 32
        args.hidden_features = 48
        args.n_components = 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    baseline_checkpoint = args.baseline_dir / "posterior_model.pt"
    baseline_dataset_path = args.baseline_dir / "training_dataset.npz"
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_checkpoint}")
    if not baseline_dataset_path.exists():
        raise FileNotFoundError(f"Missing baseline training data: {baseline_dataset_path}")

    baseline, sim_config, baseline_run_config = load_posterior(
        baseline_checkpoint, device
    )
    simulator = GalaxyFieldSimulator(sim_config)
    baseline_data = np.load(baseline_dataset_path)
    baseline_training_summaries = np.asarray(
        baseline_data["summaries_train"], dtype=np.float32
    )

    robust_run_config = RunConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_calibration=args.n_calibration,
        n_misspecification=args.n_comparison,
        n_posterior_samples=args.n_posterior_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        n_components=args.n_components,
        device=args.device,
        misspec_b2=args.stress_b2,
    )

    print("== building nuisance-marginalised training simulations ==")
    theta_train, x_train, b2_train = build_marginalised_dataset(
        simulator,
        args.n_train,
        args.seed + 1,
        args.b2_low,
        args.b2_high,
        "robust train",
    )
    print("== building nuisance-marginalised validation simulations ==")
    theta_validation, x_validation, b2_validation = build_marginalised_dataset(
        simulator,
        args.n_validation,
        args.seed + 2,
        args.b2_low,
        args.b2_high,
        "robust validation",
    )

    print("== training nuisance-marginalised posterior ==")
    robust, history = train_posterior(
        theta_train,
        x_train,
        theta_validation,
        x_validation,
        robust_run_config,
        device,
    )
    plot_training_history(history, args.output_dir / "robust_training_history.png")

    print("== calibrating under the nuisance prior ==")
    calibration = run_marginal_calibration(
        robust,
        simulator,
        args.n_calibration,
        args.n_posterior_samples,
        args.b2_low,
        args.b2_high,
        args.seed + 900_001,
    )
    plot_calibration(
        calibration["ranks"],
        calibration["coverage"],
        args.n_posterior_samples,
        args.output_dir / "robust_calibration.png",
    )

    print("== comparing baseline and nuisance-marginalised posteriors ==")
    comparison = evaluate_two_posteriors(
        baseline=baseline,
        robust=robust,
        simulator=simulator,
        baseline_training_summaries=baseline_training_summaries,
        robust_training_summaries=x_train,
        n_tests=args.n_comparison,
        n_posterior_samples=args.n_posterior_samples,
        stress_b2=args.stress_b2,
        seed=args.seed + 950_001,
    )
    plot_model_comparison(
        comparison["metrics"], args.output_dir / "baseline_vs_robust.png"
    )

    coverage_index = int(np.argmin(np.abs(CREDIBILITY_LEVELS - 0.68)))
    calibration_errors = calibration["medians"] - calibration["truths"]
    robust_metrics = {
        "simulation_config": asdict(sim_config),
        "baseline_run_config": asdict(baseline_run_config),
        "robust_run_config": asdict(robust_run_config),
        "training_nuisance_prior": {
            "b2_low": float(args.b2_low),
            "b2_high": float(args.b2_high),
        },
        "robust_calibration": {
            "coverage_at_68_percent": {
                name: float(calibration["coverage"][index, coverage_index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "posterior_median_rmse": {
                name: float(np.sqrt(np.mean(calibration_errors[:, index] ** 2)))
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "coverage_curves": {
                name: calibration["coverage"][index].tolist()
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "credibility_levels": CREDIBILITY_LEVELS.tolist(),
        },
        "baseline_vs_robust": comparison["metrics"],
        "best_validation_nll": float(min(history["validation_nll"])),
        "epochs_completed": len(history["train_nll"]),
    }
    with (args.output_dir / "robust_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(robust_metrics, handle, indent=2)

    np.savez_compressed(
        args.output_dir / "robust_training_dataset.npz",
        theta_train=theta_train,
        summaries_train=x_train,
        nonlinear_bias_b2_train=b2_train,
        theta_validation=theta_validation,
        summaries_validation=x_validation,
        nonlinear_bias_b2_validation=b2_validation,
    )
    np.savez_compressed(
        args.output_dir / "robust_diagnostics.npz",
        calibration_truths=calibration["truths"],
        calibration_b2=calibration["b2"],
        calibration_ranks=calibration["ranks"],
        calibration_coverage=calibration["coverage"],
        calibration_covered=calibration["covered"],
        calibration_medians=calibration["medians"],
        calibration_lower68=calibration["lower68"],
        calibration_upper68=calibration["upper68"],
        **comparison["arrays"],
    )
    save_robust_checkpoint(
        robust,
        sim_config,
        robust_run_config,
        args.b2_low,
        args.b2_high,
        args.output_dir / "robust_posterior_model.pt",
    )

    print("\nFixed-b2 stress-test comparison:")
    for name in PARAMETER_NAMES:
        baseline_result = comparison["metrics"]["models"]["baseline"]["b2_stress"]["parameters"][name]
        robust_result = comparison["metrics"]["models"]["robust"]["b2_stress"]["parameters"][name]
        print(
            f"  {name}: coverage {baseline_result['coverage_68']:.3f} -> "
            f"{robust_result['coverage_68']:.3f}; RMSE "
            f"{baseline_result['posterior_median_rmse']:.4f} -> "
            f"{robust_result['posterior_median_rmse']:.4f}; width "
            f"{baseline_result['median_68_width']:.4f} -> "
            f"{robust_result['median_68_width']:.4f}"
        )
    print(f"wrote robust experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
