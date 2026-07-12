#!/usr/bin/env python3
"""Calibrated neural posterior estimation for a controlled galaxy-survey toy model.

This is a compact, self-contained research prototype for testing *inference
reliability*, not a precision-cosmology analysis.  It implements

    prior over (Omega_m, sigma_8, b_1)
      -> 3-D Gaussian matter field with a BBKS-shaped linear spectrum
      -> linear redshift-space distortions
      -> biased lognormal galaxy intensity
      -> Poisson galaxy sampling (actual shot noise)
      -> P_0(k), P_2(k), and non-Gaussian count summaries
      -> conditional Gaussian-mixture neural posterior (NPE)
      -> SBC ranks, empirical coverage, and simulator-mismatch stress test.

The deliberately controlled simulator makes every approximation visible.  The
scientific question is whether an amortised posterior remains calibrated when
the observation contains a nonlinear galaxy-bias term absent from training.

Dependencies: numpy, torch, matplotlib

Examples
--------
Quick application-quality run (CPU):
    python sbi_lss_lab.py --quick --output-dir results/quick

Tiny CI/smoke run:
    python sbi_lss_lab.py --smoke --output-dir results/smoke

Larger run:
    python sbi_lss_lab.py --n-train 3000 --n-validation 400 \
        --n-calibration 150 --epochs 300 --output-dir results/full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Array = np.ndarray
SummaryKind = Literal["pk", "pk_ng"]


# -----------------------------------------------------------------------------
# Configuration and reproducibility
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    box_size: float = 500.0  # Mpc/h
    grid_size: int = 32
    n_power_bins: int = 8
    h: float = 0.6711
    n_s: float = 0.9624
    nbar_density: float = 5.0e-4  # (h/Mpc)^3
    lognormal_strength: float = 0.55
    summary_kind: SummaryKind = "pk_ng"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20260710
    n_train: int = 2000
    n_validation: int = 300
    n_calibration: int = 100
    n_misspecification: int = 30
    n_posterior_samples: int = 1000
    epochs: int = 220
    batch_size: int = 128
    learning_rate: float = 8.0e-4
    weight_decay: float = 1.0e-5
    patience: int = 30
    hidden_features: int = 128
    n_components: int = 8
    device: str = "auto"
    misspec_b2: float = 0.15


PARAMETER_NAMES = ("Omega_m", "sigma_8", "b_1")
PARAMETER_LABELS = (r"$\Omega_m$", r"$\sigma_8$", r"$b_1$")
PRIOR_LOW = np.array([0.20, 0.65, 0.80], dtype=np.float32)
PRIOR_HIGH = np.array([0.40, 0.95, 1.80], dtype=np.float32)
CREDIBILITY_LEVELS = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.68, 0.80, 0.90, 0.95],
    dtype=np.float64,
)


def set_global_seed(seed: int) -> None:
    # Small dense networks are slower when PyTorch launches dozens of CPU threads.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# 3-D cosmological toy simulator
# -----------------------------------------------------------------------------


def bbks_transfer(k: Array, omega_m: float, h: float) -> Array:
    """BBKS CDM transfer-function approximation for k in h/Mpc.

    This ignores baryonic wiggles and is intentionally labelled as a toy-model
    component.  CAMB/CLASS or simulation products should replace it for science.
    """

    q = np.maximum(k, 1.0e-10) / (omega_m * h)
    numerator = np.log1p(2.34 * q) / (2.34 * q)
    denominator = (
        1.0
        + 3.89 * q
        + (16.1 * q) ** 2
        + (5.46 * q) ** 3
        + (6.71 * q) ** 4
    ) ** 0.25
    return numerator / denominator


def top_hat_window(x: Array) -> Array:
    """Fourier-space spherical top-hat window, stable near x=0."""

    out = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > 1.0e-4
    xm = x[mask]
    out[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / xm**3
    return out


class GalaxyFieldSimulator:
    """Forward model from cosmological/nuisance parameters to summaries.

    The field generator uses FFT-filtered white noise with the discrete power
    normalisation chosen so that the *expected grid-resolved* top-hat variance at
    8 Mpc/h equals sigma_8^2.  This removes a resolution-dependent amplitude bug
    that is common in small FFT demonstrations.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        n = config.grid_size
        length = config.box_size
        dx = length / n

        k_full_axis = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        k_rfft_axis = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)

        kx, ky, kz = np.meshgrid(
            k_full_axis, k_full_axis, k_rfft_axis, indexing="ij"
        )
        self.k_rfft = np.sqrt(kx**2 + ky**2 + kz**2)
        self.mu_rfft = np.divide(
            kz,
            self.k_rfft,
            out=np.zeros_like(kz),
            where=self.k_rfft > 0,
        )

        fx, fy, fz = np.meshgrid(
            k_full_axis, k_full_axis, k_full_axis, indexing="ij"
        )
        self.k_full = np.sqrt(fx**2 + fy**2 + fz**2)
        self.mu_full = np.divide(
            fz,
            self.k_full,
            out=np.zeros_like(fz),
            where=self.k_full > 0,
        )

        k_fundamental = 2.0 * np.pi / length
        k_nyquist = np.pi / dx
        self.bin_edges = np.geomspace(
            1.5 * k_fundamental, 0.75 * k_nyquist, config.n_power_bins + 1
        )
        self.bin_masks = [
            (self.k_full >= lo) & (self.k_full < hi)
            for lo, hi in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ]
        if any(mask.sum() < 4 for mask in self.bin_masks):
            raise ValueError("Power-spectrum bins contain too few Fourier modes.")

    @property
    def summary_size(self) -> int:
        base = 2 * self.config.n_power_bins
        return base if self.config.summary_kind == "pk" else base + 4

    def _base_power(self, k: Array, omega_m: float) -> Array:
        base = np.zeros_like(k, dtype=np.float64)
        positive = k > 0
        base[positive] = (
            k[positive] ** self.config.n_s
            * bbks_transfer(k[positive], omega_m, self.config.h) ** 2
        )
        return base

    def _discrete_sigma8_amplitude(self, omega_m: float, sigma_8: float) -> float:
        base_full = self._base_power(self.k_full, omega_m)
        window = top_hat_window(8.0 * self.k_full)
        expected_variance = np.sum(base_full * window**2) / self.config.box_size**3
        if not np.isfinite(expected_variance) or expected_variance <= 0:
            raise RuntimeError("Invalid discrete sigma_8 normalisation.")
        return float(sigma_8**2 / expected_variance)

    def _matter_modes(self, omega_m: float, sigma_8: float, seed: int) -> Array:
        n = self.config.grid_size
        volume = self.config.box_size**3
        amplitude = self._discrete_sigma8_amplitude(omega_m, sigma_8)
        power = amplitude * self._base_power(self.k_rfft, omega_m)

        rng = np.random.default_rng(seed)
        white = rng.normal(size=(n, n, n))
        white_modes = np.fft.rfftn(white)
        # E|FFT(white)|^2 = N^3.  The factor below therefore gives
        # E|delta_k(physical)|^2 = V P(k) under numpy's FFT convention.
        fourier_filter = np.sqrt(n**3 * power / volume)
        modes = white_modes * fourier_filter
        modes[0, 0, 0] = 0.0
        return modes

    def _galaxy_counts(
        self,
        theta: Array,
        seed: int,
        nonlinear_bias_b2: float = 0.0,
        nbar_factor: float = 1.0,
    ) -> Array:
        omega_m, sigma_8, b1 = map(float, theta)
        modes = self._matter_modes(omega_m, sigma_8, seed)
        n = self.config.grid_size

        matter = np.fft.irfftn(modes, s=(n, n, n), axes=(0, 1, 2)).real
        growth_rate = omega_m**0.55
        rsd_kernel = b1 + growth_rate * self.mu_rfft**2
        biased_rsd = np.fft.irfftn(
            modes * rsd_kernel, s=(n, n, n), axes=(0, 1, 2)
        ).real

        exponent = self.config.lognormal_strength * biased_rsd
        if nonlinear_bias_b2 != 0.0:
            quadratic = matter**2 - np.mean(matter**2)
            quadratic_std = np.std(quadratic)
            if quadratic_std > 0:
                exponent = exponent + nonlinear_bias_b2 * quadratic / quadratic_std

        # Stable exact mean normalisation for both Gaussian and misspecified cases.
        exponent = np.clip(exponent, -12.0, 12.0)
        max_exponent = float(np.max(exponent))
        log_mean_exp = max_exponent + math.log(
            float(np.mean(np.exp(exponent - max_exponent)))
        )
        exponent = exponent - log_mean_exp

        voxel_volume = (self.config.box_size / n) ** 3
        expected_per_voxel = (
            self.config.nbar_density * nbar_factor * voxel_volume
        )
        intensity = expected_per_voxel * np.exp(exponent)
        rng = np.random.default_rng(seed + 10_000_019)
        return rng.poisson(intensity).astype(np.float64)

    def _summarise(self, counts: Array) -> Array:
        n = self.config.grid_size
        volume = self.config.box_size**3
        voxel_volume = (self.config.box_size / n) ** 3

        mean_count = float(np.mean(counts))
        if mean_count <= 0:
            raise RuntimeError("Galaxy realisation contains no objects.")
        overdensity = counts / mean_count - 1.0
        modes = np.fft.fftn(overdensity) * voxel_volume
        mode_power = np.abs(modes) ** 2 / volume

        number_density_estimate = mean_count / voxel_volume
        shot_noise = 1.0 / number_density_estimate
        legendre_2 = 0.5 * (3.0 * self.mu_full**2 - 1.0)

        p0_values: list[float] = []
        p2_values: list[float] = []
        for mask in self.bin_masks:
            p0_raw = float(np.mean(mode_power[mask]))
            p0 = max(p0_raw - shot_noise, 1.0e-3)
            p2 = float(5.0 * np.mean(mode_power[mask] * legendre_2[mask]))
            p0_values.append(math.log10(p0))
            p2_values.append(float(np.clip(p2 / max(p0_raw, 1.0e-6), -2.0, 2.0)))

        summary = p0_values + p2_values
        if self.config.summary_kind == "pk_ng":
            variance = float(np.var(overdensity))
            std = math.sqrt(max(variance, 1.0e-12))
            skewness = float(np.mean(overdensity**3) / std**3)
            fano = float(np.var(counts) / max(mean_count, 1.0e-12))
            zero_fraction = float(np.mean(counts == 0))
            clipped_skew = float(np.clip(skewness, -20.0, 20.0))
            summary.extend(
                [
                    math.log1p(variance),
                    clipped_skew,
                    math.log1p(fano),
                    zero_fraction,
                ]
            )

        result = np.asarray(summary, dtype=np.float32)
        if result.shape != (self.summary_size,) or not np.all(np.isfinite(result)):
            raise RuntimeError("Simulator produced an invalid summary vector.")
        return result

    def simulate(
        self,
        theta: Array,
        seed: int,
        nonlinear_bias_b2: float = 0.0,
        nbar_factor: float = 1.0,
    ) -> Array:
        counts = self._galaxy_counts(
            theta,
            seed=seed,
            nonlinear_bias_b2=nonlinear_bias_b2,
            nbar_factor=nbar_factor,
        )
        return self._summarise(counts)


# -----------------------------------------------------------------------------
# Training data and standardisation
# -----------------------------------------------------------------------------


def sample_prior(rng: np.random.Generator, n: int) -> Array:
    return rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n, len(PRIOR_LOW))).astype(
        np.float32
    )


def build_dataset(
    simulator: GalaxyFieldSimulator,
    n_samples: int,
    seed: int,
    label: str,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    theta = sample_prior(rng, n_samples)
    summaries = np.empty((n_samples, simulator.summary_size), dtype=np.float32)
    progress_step = max(n_samples // 10, 1)
    for index, parameters in enumerate(theta):
        summaries[index] = simulator.simulate(parameters, seed + 100_000 + index)
        if (index + 1) % progress_step == 0 or index + 1 == n_samples:
            print(f"  {label}: simulated {index + 1}/{n_samples}")
    return theta, summaries


@dataclass
class Standardizer:
    mean: Array
    std: Array

    @classmethod
    def fit(cls, values: Array) -> "Standardizer":
        mean = np.mean(values, axis=0).astype(np.float32)
        std = np.std(values, axis=0).astype(np.float32)
        std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, values: Array) -> Array:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: Array) -> Array:
        return (values * self.std + self.mean).astype(np.float32)


@dataclass
class BoundedParameterTransform:
    """Map bounded physical parameters to an unconstrained standard space.

    A logit transform builds the uniform-prior support into posterior sampling,
    avoiding ad-hoc clipping or fragile rejection sampling for OOD summaries.
    """

    low: Array
    high: Array
    latent_scaler: Standardizer

    @classmethod
    def fit(cls, values: Array, low: Array, high: Array) -> "BoundedParameterTransform":
        fraction = (values - low[None, :]) / (high - low)[None, :]
        fraction = np.clip(fraction, 1.0e-5, 1.0 - 1.0e-5)
        latent = np.log(fraction / (1.0 - fraction)).astype(np.float32)
        return cls(low=low.copy(), high=high.copy(), latent_scaler=Standardizer.fit(latent))

    def transform(self, values: Array) -> Array:
        fraction = (values - self.low[None, :]) / (self.high - self.low)[None, :]
        fraction = np.clip(fraction, 1.0e-5, 1.0 - 1.0e-5)
        latent = np.log(fraction / (1.0 - fraction)).astype(np.float32)
        return self.latent_scaler.transform(latent)

    def inverse(self, values: Array) -> Array:
        latent = self.latent_scaler.inverse(values)
        latent = np.clip(latent, -30.0, 30.0)
        fraction = 1.0 / (1.0 + np.exp(-latent))
        return (self.low[None, :] + fraction * (self.high - self.low)[None, :]).astype(np.float32)


# -----------------------------------------------------------------------------
# Conditional Gaussian-mixture neural posterior
# -----------------------------------------------------------------------------


class MixturePosteriorNetwork(nn.Module):
    """Diagonal Gaussian-mixture estimator for q(theta | summary)."""

    def __init__(
        self,
        input_features: int,
        parameter_features: int,
        hidden_features: int,
        n_components: int,
    ) -> None:
        super().__init__()
        self.parameter_features = parameter_features
        self.n_components = n_components
        self.backbone = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )
        self.logit_head = nn.Linear(hidden_features, n_components)
        self.mean_head = nn.Linear(
            hidden_features, n_components * parameter_features
        )
        self.log_scale_head = nn.Linear(
            hidden_features, n_components * parameter_features
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.backbone(x)
        logits = self.logit_head(hidden)
        means = self.mean_head(hidden).view(
            -1, self.n_components, self.parameter_features
        )
        log_scales = self.log_scale_head(hidden).view(
            -1, self.n_components, self.parameter_features
        )
        log_scales = torch.clamp(log_scales, min=-5.0, max=2.0)
        return logits, means, log_scales

    def negative_log_likelihood(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        logits, means, log_scales = self(x)
        centred = (theta[:, None, :] - means) / torch.exp(log_scales)
        component_log_prob = -0.5 * (
            centred**2 + 2.0 * log_scales + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        mixture_log_prob = torch.log_softmax(logits, dim=-1) + component_log_prob
        return -torch.logsumexp(mixture_log_prob, dim=-1).mean()


@dataclass
class TrainedPosterior:
    model: MixturePosteriorNetwork
    x_scaler: Standardizer
    theta_scaler: BoundedParameterTransform
    device: torch.device

    @torch.no_grad()
    def sample(
        self,
        summary: Array,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Array:
        self.model.eval()
        x_standard = self.x_scaler.transform(np.asarray(summary)[None, :])
        x_tensor = torch.from_numpy(x_standard).to(self.device)
        logits, means, log_scales = self.model(x_tensor)
        probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy()
        means_np = means[0].cpu().numpy()
        scales_np = np.exp(log_scales[0].cpu().numpy())

        components = rng.choice(len(probabilities), size=n_samples, p=probabilities)
        standard_samples = rng.normal(
            loc=means_np[components], scale=scales_np[components]
        ).astype(np.float32)
        return self.theta_scaler.inverse(standard_samples)


def train_posterior(
    theta_train: Array,
    x_train: Array,
    theta_validation: Array,
    x_validation: Array,
    config: RunConfig,
    device: torch.device,
) -> tuple[TrainedPosterior, dict[str, list[float]]]:
    x_scaler = Standardizer.fit(x_train)
    theta_scaler = BoundedParameterTransform.fit(theta_train, PRIOR_LOW, PRIOR_HIGH)

    x_train_t = torch.from_numpy(x_scaler.transform(x_train))
    theta_train_t = torch.from_numpy(theta_scaler.transform(theta_train))
    x_val_t = torch.from_numpy(x_scaler.transform(x_validation)).to(device)
    theta_val_t = torch.from_numpy(theta_scaler.transform(theta_validation)).to(
        device
    )

    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x_train_t, theta_train_t),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    model = MixturePosteriorNetwork(
        input_features=x_train.shape[1],
        parameter_features=theta_train.shape[1],
        hidden_features=config.hidden_features,
        n_components=config.n_components,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    epochs_without_improvement = 0
    history = {"train_nll": [], "validation_nll": []}

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for x_batch, theta_batch in loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(x_batch, theta_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            total_loss += float(loss.detach().cpu()) * len(x_batch)
            total_examples += len(x_batch)

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                model.negative_log_likelihood(x_val_t, theta_val_t).cpu()
            )
        training_loss = total_loss / max(total_examples, 1)
        history["train_nll"].append(training_loss)
        history["validation_nll"].append(validation_loss)

        improved = validation_loss < best_validation - 1.0e-4
        if improved:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"  epoch {epoch + 1:4d}: train NLL={training_loss:.4f}, "
                f"validation NLL={validation_loss:.4f}"
            )
        if epochs_without_improvement >= config.patience:
            print(f"  early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    model.to(device)
    return (
        TrainedPosterior(
            model=model,
            x_scaler=x_scaler,
            theta_scaler=theta_scaler,
            device=device,
        ),
        history,
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def central_interval(samples: Array, level: float) -> tuple[Array, Array]:
    tail = 0.5 * (1.0 - level)
    return (
        np.quantile(samples, tail, axis=0),
        np.quantile(samples, 1.0 - tail, axis=0),
    )


def run_calibration(
    posterior: TrainedPosterior,
    simulator: GalaxyFieldSimulator,
    n_tests: int,
    n_posterior_samples: int,
    seed: int,
) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    truths = sample_prior(rng, n_tests)
    ranks = np.zeros((n_tests, len(PARAMETER_NAMES)), dtype=np.int64)
    covered = np.zeros(
        (n_tests, len(PARAMETER_NAMES), len(CREDIBILITY_LEVELS)),
        dtype=np.float64,
    )
    posterior_medians = np.zeros_like(truths)

    for index, truth in enumerate(truths):
        summary = simulator.simulate(truth, seed + 200_000 + index)
        samples = posterior.sample(
            summary, n_posterior_samples, rng=np.random.default_rng(seed + index)
        )
        ranks[index] = np.sum(samples < truth[None, :], axis=0)
        posterior_medians[index] = np.median(samples, axis=0)
        for level_index, level in enumerate(CREDIBILITY_LEVELS):
            lower, upper = central_interval(samples, float(level))
            covered[index, :, level_index] = (truth >= lower) & (truth <= upper)
        if (index + 1) % max(n_tests // 5, 1) == 0 or index + 1 == n_tests:
            print(f"  calibration: completed {index + 1}/{n_tests}")

    coverage = np.mean(covered, axis=0)
    return ranks, coverage, posterior_medians - truths


def summary_ood_diagnostics(
    summary: Array,
    training_summaries: Array,
    scaler: Standardizer,
) -> dict[str, float]:
    """Simple feature-space OOD diagnostics for a candidate observation.

    These are not a calibrated hypothesis test.  They are warning indicators
    showing whether an observation lies far from the simulator support used for
    training the neural posterior.
    """

    train_z = scaler.transform(training_summaries).astype(np.float64)
    summary_z = scaler.transform(np.asarray(summary)[None, :])[0].astype(np.float64)
    covariance = np.cov(train_z, rowvar=False)
    regularisation = 0.05 * np.eye(covariance.shape[0])
    inverse_covariance = np.linalg.pinv(covariance + regularisation)
    mahalanobis_rms = float(
        np.sqrt(max(summary_z @ inverse_covariance @ summary_z, 0.0) / len(summary_z))
    )
    nearest_rms = float(
        np.min(np.sqrt(np.mean((train_z - summary_z[None, :]) ** 2, axis=1)))
    )
    return {
        "mahalanobis_rms": mahalanobis_rms,
        "nearest_training_rms": nearest_rms,
        "maximum_absolute_z": float(np.max(np.abs(summary_z))),
    }


def run_paired_misspecification_suite(
    posterior: TrainedPosterior,
    simulator: GalaxyFieldSimulator,
    training_summaries: Array,
    n_tests: int,
    n_posterior_samples: int,
    nonlinear_bias_b2: float,
    seed: int,
) -> dict[str, object]:
    """Compare well-specified and misspecified posteriors on paired universes."""

    rng = np.random.default_rng(seed)
    truths = sample_prior(rng, n_tests)
    good_errors = np.zeros_like(truths, dtype=np.float64)
    bad_errors = np.zeros_like(truths, dtype=np.float64)
    good_covered = np.zeros_like(truths, dtype=bool)
    bad_covered = np.zeros_like(truths, dtype=bool)
    good_widths = np.zeros_like(truths, dtype=np.float64)
    bad_widths = np.zeros_like(truths, dtype=np.float64)
    good_ood = np.zeros(n_tests, dtype=np.float64)
    bad_ood = np.zeros(n_tests, dtype=np.float64)

    for index, truth in enumerate(truths):
        simulation_seed = seed + 300_000 + index
        good_summary = simulator.simulate(truth, simulation_seed)
        bad_summary = simulator.simulate(
            truth, simulation_seed, nonlinear_bias_b2=nonlinear_bias_b2
        )
        good_samples = posterior.sample(
            good_summary,
            n_posterior_samples,
            np.random.default_rng(seed + 400_000 + index),
        )
        bad_samples = posterior.sample(
            bad_summary,
            n_posterior_samples,
            np.random.default_rng(seed + 500_000 + index),
        )
        good_median = np.median(good_samples, axis=0)
        bad_median = np.median(bad_samples, axis=0)
        good_errors[index] = good_median - truth
        bad_errors[index] = bad_median - truth

        good_lower, good_upper = central_interval(good_samples, 0.68)
        bad_lower, bad_upper = central_interval(bad_samples, 0.68)
        good_covered[index] = (truth >= good_lower) & (truth <= good_upper)
        bad_covered[index] = (truth >= bad_lower) & (truth <= bad_upper)
        good_widths[index] = good_upper - good_lower
        bad_widths[index] = bad_upper - bad_lower

        good_ood[index] = summary_ood_diagnostics(
            good_summary, training_summaries, posterior.x_scaler
        )["mahalanobis_rms"]
        bad_ood[index] = summary_ood_diagnostics(
            bad_summary, training_summaries, posterior.x_scaler
        )["mahalanobis_rms"]

        if (index + 1) % max(n_tests // 5, 1) == 0 or index + 1 == n_tests:
            print(f"  paired mismatch: completed {index + 1}/{n_tests}")

    output: dict[str, object] = {
        "n_tests": int(n_tests),
        "nonlinear_bias_b2": float(nonlinear_bias_b2),
        "well_specified": {},
        "misspecified": {},
        "paired_change": {},
        "ood_mahalanobis_rms": {
            "well_specified_median": float(np.median(good_ood)),
            "misspecified_median": float(np.median(bad_ood)),
            "well_specified_90th_percentile": float(np.quantile(good_ood, 0.9)),
            "misspecified_90th_percentile": float(np.quantile(bad_ood, 0.9)),
        },
    }
    for parameter_index, name in enumerate(PARAMETER_NAMES):
        good_rmse = float(np.sqrt(np.mean(good_errors[:, parameter_index] ** 2)))
        bad_rmse = float(np.sqrt(np.mean(bad_errors[:, parameter_index] ** 2)))
        output["well_specified"][name] = {
            "median_bias": float(np.median(good_errors[:, parameter_index])),
            "posterior_median_rmse": good_rmse,
            "coverage_68": float(np.mean(good_covered[:, parameter_index])),
            "median_68_width": float(np.median(good_widths[:, parameter_index])),
        }
        output["misspecified"][name] = {
            "median_bias": float(np.median(bad_errors[:, parameter_index])),
            "posterior_median_rmse": bad_rmse,
            "coverage_68": float(np.mean(bad_covered[:, parameter_index])),
            "median_68_width": float(np.median(bad_widths[:, parameter_index])),
        }
        output["paired_change"][name] = {
            "rmse_ratio": bad_rmse / max(good_rmse, 1.0e-12),
            "coverage_change": float(
                np.mean(bad_covered[:, parameter_index])
                - np.mean(good_covered[:, parameter_index])
            ),
            "median_width_ratio": float(
                np.median(bad_widths[:, parameter_index])
                / max(np.median(good_widths[:, parameter_index]), 1.0e-12)
            ),
            "median_absolute_posterior_shift": float(
                np.median(
                    np.abs(
                        bad_errors[:, parameter_index]
                        - good_errors[:, parameter_index]
                    )
                )
            ),
        }
    return output


def posterior_summary(samples: Array, truth: Array) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        lower, median, upper = np.quantile(samples[:, index], [0.16, 0.5, 0.84])
        output[name] = {
            "truth": float(truth[index]),
            "median": float(median),
            "lower_16": float(lower),
            "upper_84": float(upper),
            "bias": float(median - truth[index]),
        }
    return output


def plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(history["train_nll"], label="training")
    axis.plot(history["validation_nll"], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("negative log likelihood")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_corner(samples: Array, truth: Array, path: Path) -> None:
    dimensions = samples.shape[1]
    fig, axes = plt.subplots(
        dimensions, dimensions, figsize=(8.0, 8.0), squeeze=False
    )
    for row in range(dimensions):
        for column in range(dimensions):
            axis = axes[row, column]
            if row == column:
                axis.hist(samples[:, column], bins=35, density=True, alpha=0.8)
                axis.axvline(truth[column], linestyle="--")
            elif row > column:
                thin = max(len(samples) // 2500, 1)
                axis.scatter(
                    samples[::thin, column],
                    samples[::thin, row],
                    s=3,
                    alpha=0.2,
                    rasterized=True,
                )
                axis.plot(truth[column], truth[row], marker="*", markersize=10)
            else:
                axis.axis("off")
                continue
            if row == dimensions - 1:
                axis.set_xlabel(PARAMETER_LABELS[column])
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(PARAMETER_LABELS[row])
            elif column > 0:
                axis.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_calibration(ranks: Array, coverage: Array, n_post: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))
    bins = 10
    expected = len(ranks) / bins
    for index, label in enumerate(PARAMETER_LABELS):
        axes[index].hist(ranks[:, index], bins=bins, range=(0, n_post))
        axes[index].axhline(expected, linestyle="--")
        axes[index].set_title(f"SBC ranks: {label}")
        axes[index].set_xlabel("posterior rank")
    for index, label in enumerate(PARAMETER_LABELS):
        axes[3].plot(CREDIBILITY_LEVELS, coverage[index], marker="o", label=label)
    axes[3].plot([0, 1], [0, 1], linestyle=":")
    nominal = np.linspace(0.0, 1.0, 200)
    standard_error = np.sqrt(nominal * (1.0 - nominal) / max(len(ranks), 1))
    lower_band = np.clip(nominal - 1.96 * standard_error, 0.0, 1.0)
    upper_band = np.clip(nominal + 1.96 * standard_error, 0.0, 1.0)
    axes[3].fill_between(nominal, lower_band, upper_band, alpha=0.15, label="95% binomial band")
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel("nominal credibility")
    axes[3].set_ylabel("empirical coverage")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_misspecification(
    well_specified: Array,
    misspecified: Array,
    truth: Array,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 5.0))
    thin_good = max(len(well_specified) // 1800, 1)
    thin_bad = max(len(misspecified) // 1800, 1)
    axis.scatter(
        well_specified[::thin_good, 0],
        well_specified[::thin_good, 1],
        s=5,
        alpha=0.25,
        label="well-specified",
    )
    axis.scatter(
        misspecified[::thin_bad, 0],
        misspecified[::thin_bad, 1],
        s=5,
        alpha=0.25,
        label="unmodelled nonlinear bias",
    )
    axis.plot(truth[0], truth[1], marker="*", markersize=13, label="truth")
    axis.set_xlabel(PARAMETER_LABELS[0])
    axis.set_ylabel(PARAMETER_LABELS[1])
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_example_summary(
    simulator: GalaxyFieldSimulator, theta: Array, seed: int, path: Path
) -> None:
    summary = simulator.simulate(theta, seed)
    n_bins = simulator.config.n_power_bins
    centres = np.sqrt(simulator.bin_edges[:-1] * simulator.bin_edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(centres, summary[:n_bins], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$\log_{10} P_0(k)$")
    axes[1].plot(centres, summary[n_bins : 2 * n_bins], marker="o")
    axes[1].axhline(0.0, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    axes[1].set_ylabel(r"$P_2/P_{0,\mathrm{raw}}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# End-to-end run
# -----------------------------------------------------------------------------


def run(sim_config: SimulationConfig, run_config: RunConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(run_config.seed)
    device = resolve_device(run_config.device)
    print(f"device: {device}")
    print(f"simulation config: {sim_config}")
    print(f"run config: {run_config}")

    simulator = GalaxyFieldSimulator(sim_config)
    print("== building training simulations ==")
    theta_train, x_train = build_dataset(
        simulator, run_config.n_train, run_config.seed + 1, "train"
    )
    print("== building validation simulations ==")
    theta_validation, x_validation = build_dataset(
        simulator, run_config.n_validation, run_config.seed + 2, "validation"
    )

    print("== training conditional mixture posterior ==")
    posterior, history = train_posterior(
        theta_train,
        x_train,
        theta_validation,
        x_validation,
        config=run_config,
        device=device,
    )
    plot_training_history(history, output_dir / "training_history.png")

    truth = np.array([0.315, 0.811, 1.25], dtype=np.float32)
    observation = simulator.simulate(truth, run_config.seed + 700_001)
    posterior_rng = np.random.default_rng(run_config.seed + 800_001)
    samples = posterior.sample(
        observation, run_config.n_posterior_samples * 4, posterior_rng
    )
    plot_corner(samples, truth, output_dir / "posterior_corner.png")
    plot_example_summary(
        simulator, truth, run_config.seed + 700_001, output_dir / "summary_example.png"
    )

    print("posterior for held-out mock (median and central 68% interval):")
    well_summary = posterior_summary(samples, truth)
    for name in PARAMETER_NAMES:
        values = well_summary[name]
        print(
            f"  {name}: {values['median']:.4f} "
            f"[{values['lower_16']:.4f}, {values['upper_84']:.4f}] "
            f"truth={values['truth']:.4f}"
        )

    print("== simulation-based calibration ==")
    ranks, coverage, calibration_errors = run_calibration(
        posterior,
        simulator,
        n_tests=run_config.n_calibration,
        n_posterior_samples=run_config.n_posterior_samples,
        seed=run_config.seed + 900_001,
    )
    plot_calibration(
        ranks,
        coverage,
        run_config.n_posterior_samples,
        output_dir / "calibration.png",
    )

    print("== simulator-mismatch stress test ==")
    misspecified_observation = simulator.simulate(
        truth,
        run_config.seed + 700_001,
        nonlinear_bias_b2=run_config.misspec_b2,
    )
    misspecified_samples = posterior.sample(
        misspecified_observation,
        run_config.n_posterior_samples * 4,
        np.random.default_rng(run_config.seed + 800_002),
    )
    plot_misspecification(
        samples,
        misspecified_samples,
        truth,
        output_dir / "misspecification.png",
    )
    misspecified_summary = posterior_summary(misspecified_samples, truth)
    ood_diagnostics = {
        "well_specified": summary_ood_diagnostics(
            observation, x_train, posterior.x_scaler
        ),
        "misspecified": summary_ood_diagnostics(
            misspecified_observation, x_train, posterior.x_scaler
        ),
    }
    print(
        "  summary-space OOD score (Mahalanobis RMS): "
        f"well-specified={ood_diagnostics['well_specified']['mahalanobis_rms']:.2f}, "
        f"misspecified={ood_diagnostics['misspecified']['mahalanobis_rms']:.2f}"
    )

    misspecification_diagnostics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        good_median = float(np.median(samples[:, index]))
        bad_median = float(np.median(misspecified_samples[:, index]))
        good_width = float(
            np.quantile(samples[:, index], 0.84)
            - np.quantile(samples[:, index], 0.16)
        )
        bad_width = float(
            np.quantile(misspecified_samples[:, index], 0.84)
            - np.quantile(misspecified_samples[:, index], 0.16)
        )
        misspecification_diagnostics[name] = {
            "median_shift": bad_median - good_median,
            "shift_in_good_half_widths": (bad_median - good_median)
            / max(0.5 * good_width, 1.0e-8),
            "central_68_width_ratio": bad_width / max(good_width, 1.0e-8),
        }
        values = misspecification_diagnostics[name]
        print(
            f"  {name}: median shift={values['median_shift']:+.4f}, "
            f"shift={values['shift_in_good_half_widths']:+.2f} posterior half-widths, "
            f"width ratio={values['central_68_width_ratio']:.2f}"
        )

    print("== paired simulator-mismatch suite ==")
    paired_misspecification = run_paired_misspecification_suite(
        posterior=posterior,
        simulator=simulator,
        training_summaries=x_train,
        n_tests=run_config.n_misspecification,
        n_posterior_samples=run_config.n_posterior_samples,
        nonlinear_bias_b2=run_config.misspec_b2,
        seed=run_config.seed + 950_001,
    )
    for name in PARAMETER_NAMES:
        good = paired_misspecification["well_specified"][name]
        bad = paired_misspecification["misspecified"][name]
        print(
            f"  {name}: 68% coverage {good['coverage_68']:.2f} -> "
            f"{bad['coverage_68']:.2f}; median-RMSE "
            f"{good['posterior_median_rmse']:.4f} -> "
            f"{bad['posterior_median_rmse']:.4f}"
        )

    calibration_rmse = np.sqrt(np.mean(calibration_errors**2, axis=0))
    coverage_at_68_index = int(np.argmin(np.abs(CREDIBILITY_LEVELS - 0.68)))
    coverage_deviation = coverage - CREDIBILITY_LEVELS[None, :]
    metrics = {
        "simulation_config": asdict(sim_config),
        "run_config": asdict(run_config),
        "device": str(device),
        "held_out_posterior": well_summary,
        "misspecified_posterior": misspecified_summary,
        "misspecification_diagnostics": misspecification_diagnostics,
        "summary_ood_diagnostics": ood_diagnostics,
        "paired_misspecification_suite": paired_misspecification,
        "calibration": {
            "credibility_levels": CREDIBILITY_LEVELS.tolist(),
            "coverage": {
                name: coverage[index].tolist()
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "posterior_median_rmse": {
                name: float(calibration_rmse[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "coverage_at_68_percent": {
                name: float(coverage[index, coverage_at_68_index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "coverage_rmse": {
                name: float(np.sqrt(np.mean(coverage_deviation[index] ** 2)))
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "maximum_absolute_coverage_error": {
                name: float(np.max(np.abs(coverage_deviation[index])))
                for index, name in enumerate(PARAMETER_NAMES)
            },
        },
        "best_validation_nll": float(min(history["validation_nll"])),
        "epochs_completed": len(history["train_nll"]),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    np.savez_compressed(
        output_dir / "training_dataset.npz",
        theta_train=theta_train,
        summaries_train=x_train,
        theta_validation=theta_validation,
        summaries_validation=x_validation,
    )
    np.savez_compressed(
        output_dir / "diagnostics.npz",
        truth=truth,
        posterior_samples=samples,
        misspecified_posterior_samples=misspecified_samples,
        ranks=ranks,
        coverage=coverage,
        calibration_errors=calibration_errors,
        bin_edges=simulator.bin_edges,
    )
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
        },
        output_dir / "posterior_model.pt",
    )
    print(f"wrote diagnostics to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated neural posterior estimation for a 3-D galaxy-field toy model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--grid-size", type=int, default=SimulationConfig.grid_size)
    parser.add_argument("--n-power-bins", type=int, default=SimulationConfig.n_power_bins)
    parser.add_argument("--summary", choices=("pk", "pk_ng"), default="pk_ng")
    parser.add_argument("--n-train", type=int, default=RunConfig.n_train)
    parser.add_argument("--n-validation", type=int, default=RunConfig.n_validation)
    parser.add_argument("--n-calibration", type=int, default=RunConfig.n_calibration)
    parser.add_argument(
        "--n-misspecification", type=int, default=RunConfig.n_misspecification
    )
    parser.add_argument(
        "--n-posterior-samples", type=int, default=RunConfig.n_posterior_samples
    )
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--hidden-features", type=int, default=RunConfig.hidden_features)
    parser.add_argument("--n-components", type=int, default=RunConfig.n_components)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--misspec-b2", type=float, default=RunConfig.misspec_b2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick and args.smoke:
        raise SystemExit("Choose only one of --quick or --smoke.")

    if args.smoke:
        args.grid_size = 16
        args.n_power_bins = 5
        args.n_train = 80
        args.n_validation = 20
        args.n_calibration = 5
        args.n_misspecification = 3
        args.n_posterior_samples = 80
        args.epochs = 5
        args.batch_size = 32
        args.hidden_features = 48
        args.n_components = 3
    elif args.quick:
        args.grid_size = 24
        args.n_train = 600
        args.n_validation = 100
        args.n_calibration = 40
        args.n_misspecification = 20
        args.n_posterior_samples = 500
        args.epochs = 100
        args.batch_size = 96
        args.hidden_features = 96
        args.n_components = 6

    simulation_config = SimulationConfig(
        grid_size=args.grid_size,
        n_power_bins=args.n_power_bins,
        summary_kind=args.summary,
    )
    run_config = RunConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_calibration=args.n_calibration,
        n_misspecification=args.n_misspecification,
        n_posterior_samples=args.n_posterior_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_features=args.hidden_features,
        n_components=args.n_components,
        device=args.device,
        misspec_b2=args.misspec_b2,
    )
    run(simulation_config, run_config, args.output_dir)


if __name__ == "__main__":
    main()
