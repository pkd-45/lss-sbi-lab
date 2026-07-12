"""Fast scientific and software checks for sbi_lss_lab.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sbi_lss_lab import (  # noqa: E402
    BoundedParameterTransform,
    GalaxyFieldSimulator,
    MixturePosteriorNetwork,
    PRIOR_HIGH,
    PRIOR_LOW,
    SimulationConfig,
    Standardizer,
    top_hat_window,
)


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = GalaxyFieldSimulator(
            SimulationConfig(grid_size=16, n_power_bins=5, summary_kind="pk_ng")
        )
        self.theta = np.array([0.31, 0.81, 1.2], dtype=np.float32)

    def test_summary_shape_and_finiteness(self) -> None:
        summary = self.simulator.simulate(self.theta, seed=123)
        self.assertEqual(summary.shape, (self.simulator.summary_size,))
        self.assertTrue(np.all(np.isfinite(summary)))

    def test_nonlinear_bias_changes_summary(self) -> None:
        baseline = self.simulator.simulate(self.theta, seed=123)
        mismatch = self.simulator.simulate(
            self.theta, seed=123, nonlinear_bias_b2=0.15
        )
        self.assertGreater(float(np.linalg.norm(baseline - mismatch)), 0.05)

    def test_discrete_sigma8_normalisation(self) -> None:
        # The expectation, averaged over random phases, should recover the
        # requested grid-resolved top-hat RMS to within a modest Monte-Carlo tolerance.
        target = float(self.theta[1])
        recovered = []
        n = self.simulator.config.grid_size
        for seed in range(12):
            modes = self.simulator._matter_modes(  # noqa: SLF001
                float(self.theta[0]), target, seed
            )
            smoothed = np.fft.irfftn(
                modes * top_hat_window(8.0 * self.simulator.k_rfft),
                s=(n, n, n),
                axes=(0, 1, 2),
            ).real
            recovered.append(float(np.std(smoothed)))
        self.assertAlmostEqual(float(np.mean(recovered)), target, delta=0.06)


class TransformAndNetworkTests(unittest.TestCase):
    def test_bounded_transform_round_trip(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(200, 3)).astype(np.float32)
        transform = BoundedParameterTransform.fit(values, PRIOR_LOW, PRIOR_HIGH)
        reconstructed = transform.inverse(transform.transform(values))
        np.testing.assert_allclose(reconstructed, values, rtol=2e-5, atol=2e-5)
        self.assertTrue(np.all(reconstructed >= PRIOR_LOW))
        self.assertTrue(np.all(reconstructed <= PRIOR_HIGH))

    def test_network_loss_is_finite(self) -> None:
        network = MixturePosteriorNetwork(
            input_features=14,
            parameter_features=3,
            hidden_features=32,
            n_components=4,
        )
        x = torch.randn(10, 14)
        theta = torch.randn(10, 3)
        loss = network.negative_log_likelihood(x, theta)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

    def test_standardizer_round_trip(self) -> None:
        rng = np.random.default_rng(9)
        values = rng.normal(size=(100, 6)).astype(np.float32)
        scaler = Standardizer.fit(values)
        np.testing.assert_allclose(
            scaler.inverse(scaler.transform(values)), values, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
