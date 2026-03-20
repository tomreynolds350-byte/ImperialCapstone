from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_DIR = REPO_ROOT / "execution"
if str(EXECUTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTION_DIR))

import bo_core  # noqa: E402
import propose_gp_candidates  # noqa: E402
import propose_round_06_candidates  # noqa: E402


class TestBayesianOptimizationCore(unittest.TestCase):
    def test_acquisition_functions_use_maximize_direction(self) -> None:
        mu = np.array([0.2, 1.2], dtype=float)
        sigma = np.array([0.1, 0.1], dtype=float)

        ei = bo_core.expected_improvement(mu, sigma, best_y=0.5, xi=0.0)
        ucb = bo_core.upper_confidence_bound(mu, sigma, kappa=1.96)

        self.assertGreater(float(ei[1]), float(ei[0]))
        self.assertGreater(float(ucb[1]), float(ucb[0]))

    def test_near_constant_targets_have_finite_gp_diagnostics(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.random((8, 2))
        y = np.array([0.0, 0.0, 0.0, -1e-4, 2e-4, 1e-4, 0.0, -1e-4], dtype=float)

        _, info = bo_core.fit_gp_model(x, y, random_state=13, n_restarts_optimizer=1)

        self.assertEqual(info["objective_direction"], "maximize")
        self.assertTrue(math.isfinite(info["best_score"]))
        self.assertTrue(math.isfinite(info["loo_mae"]))
        self.assertTrue(math.isfinite(info["loo_rmse"]))
        self.assertTrue(math.isfinite(info["target_std"]))
        self.assertGreater(info["best_score"], -1e6)

    def test_gp_fit_reports_per_dimension_length_scales(self) -> None:
        grid = np.linspace(0.05, 0.95, 12)
        x = np.column_stack([grid, grid[::-1], np.roll(grid, 3)])
        y = np.sin(2.0 * np.pi * x[:, 0]) + 0.5 * x[:, 1] - 0.25 * x[:, 2]

        _, info = bo_core.fit_gp_model(x, y, random_state=21, n_restarts_optimizer=2)

        self.assertEqual(len(info["best_length_scales"]), 3)
        self.assertIsInstance(info["length_scale_at_lower_bound"], bool)
        self.assertIsInstance(info["length_scale_at_upper_bound"], bool)
        self.assertLess(sum(np.isclose(info["best_length_scales"], bo_core.LENGTH_SCALE_BOUNDS[0], atol=1e-4)), 3)

    def test_balanced_mode_avoids_exact_boundary_candidates(self) -> None:
        x = np.load(REPO_ROOT / "initial_data" / "function_5" / "initial_inputs.npy")
        y = np.load(REPO_ROOT / "initial_data" / "function_5" / "initial_outputs.npy").reshape(-1)

        candidate, info = bo_core.choose_hybrid_candidate(
            x=x,
            y=y,
            rng=np.random.default_rng(2026),
            low=bo_core.DEFAULT_LOW,
            high=bo_core.DEFAULT_HIGH,
            boundary_margin=0.035,
            seed=2026,
            strategy="balanced",
            kappa=1.96,
            gp_restarts=1,
        )

        self.assertEqual(candidate.shape[0], x.shape[1])
        self.assertTrue(
            float(info["chosen_candidate_bound_dist"]) > 0.0 or bool(info["boundary_override_used"])
        )

    def test_write_outputs_uses_prefix_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            raw_vectors = {f"function_{i}": [0.1] * (2 if i <= 2 else 3 if i == 3 else 4 if i <= 5 else 5 if i == 6 else 6 if i == 7 else 8) for i in range(1, 9)}
            portal_strings = {key: "-".join(f"{v:.6f}" for v in value) for key, value in raw_vectors.items()}
            debug_info = {"function_1": {"objective_direction": "maximize"}}

            bo_core.write_submission_outputs(
                out_dir,
                raw_vectors,
                portal_strings,
                debug_info,
                prefix="demo_round",
                debug_label="hybrid_debug",
            )

            self.assertTrue((out_dir / "demo_round_portal_strings.txt").exists())
            self.assertTrue((out_dir / "demo_round_portal_strings.json").exists())
            self.assertTrue((out_dir / "demo_round_inputs.txt").exists())
            self.assertTrue((out_dir / "demo_round_hybrid_debug.json").exists())

    def test_wrappers_delegate_to_shared_runner(self) -> None:
        with mock.patch.object(propose_gp_candidates, "run_gp_candidate_script") as gp_runner:
            with mock.patch.object(sys, "argv", ["prog", "--prefix", "round_02_test"]):
                propose_gp_candidates.main()
            gp_runner.assert_called_once()
            gp_args = gp_runner.call_args[0][0]
            self.assertEqual(gp_args.prefix, "round_02_test")

        with mock.patch.object(propose_round_06_candidates, "run_round_candidate_script") as round_runner:
            with mock.patch.object(sys, "argv", ["prog", "--prefix", "round_06_test"]):
                propose_round_06_candidates.main()
            round_runner.assert_called_once()
            round_args = round_runner.call_args[0][0]
            self.assertEqual(round_args.prefix, "round_06_test")
            self.assertEqual(round_runner.call_args.kwargs["snapshot_filename"], "round_05_outputs_canonical.txt")

    def test_dry_run_current_initial_data_has_finite_diagnostics(self) -> None:
        rng = np.random.default_rng(99)
        for func_id in range(1, 9):
            func_dir = REPO_ROOT / "initial_data" / f"function_{func_id}"
            x = np.load(func_dir / "initial_inputs.npy")
            y = np.load(func_dir / "initial_outputs.npy").reshape(-1)

            candidate, info = bo_core.choose_hybrid_candidate(
                x=x,
                y=y,
                rng=rng,
                low=bo_core.DEFAULT_LOW,
                high=bo_core.DEFAULT_HIGH,
                boundary_margin=0.035,
                seed=3000 + func_id,
                strategy="balanced",
                kappa=1.96,
                gp_restarts=1,
            )

            self.assertEqual(candidate.shape[0], x.shape[1])
            self.assertTrue(np.all(candidate >= bo_core.DEFAULT_LOW))
            self.assertTrue(np.all(candidate <= bo_core.DEFAULT_HIGH))

            for field in (
                "best_score",
                "target_std",
                "loo_mae",
                "loo_rmse",
                "chosen_candidate_ei",
                "chosen_candidate_ucb",
                "chosen_candidate_min_dist",
                "chosen_candidate_bound_dist",
                "linear_r2_cv_mean",
                "svr_r2_cv_mean",
                "mlp_r2_cv_mean",
            ):
                self.assertTrue(math.isfinite(float(info[field])), f"{field} not finite for function_{func_id}")


if __name__ == "__main__":
    unittest.main()
