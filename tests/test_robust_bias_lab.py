import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from robust_bias_lab import build_marginalised_dataset, load_posterior
from sbi_lss_lab import GalaxyFieldSimulator, SimulationConfig


class RobustBiasDatasetTests(unittest.TestCase):
    def test_marginalised_dataset_records_nuisance(self):
        simulator = GalaxyFieldSimulator(
            SimulationConfig(grid_size=16, n_power_bins=5)
        )
        theta, summaries, b2 = build_marginalised_dataset(
            simulator,
            n_samples=6,
            seed=123,
            b2_low=-0.1,
            b2_high=0.2,
            label="test",
        )
        self.assertEqual(theta.shape, (6, 3))
        self.assertEqual(summaries.shape, (6, simulator.summary_size))
        self.assertEqual(b2.shape, (6,))
        self.assertTrue(np.all(b2 >= -0.1))
        self.assertTrue(np.all(b2 <= 0.2))
        self.assertTrue(np.all(np.isfinite(summaries)))

    def test_load_reference_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "results" / "reference_run" / "posterior_model.pt"
        if not checkpoint.exists():
            self.skipTest("reference checkpoint not available")
        posterior, sim_config, run_config = load_posterior(
            checkpoint, torch.device("cpu")
        )
        self.assertEqual(sim_config.grid_size, 32)
        self.assertGreater(run_config.n_components, 0)
        self.assertEqual(posterior.x_scaler.mean.ndim, 1)


if __name__ == "__main__":
    unittest.main()
