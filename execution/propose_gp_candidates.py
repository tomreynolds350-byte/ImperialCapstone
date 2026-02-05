from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import loguniform, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import RandomizedSearchCV


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "initial_data"


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = improvement * cdf + sigma * pdf
    ei[sigma <= 1e-9] = 0.0
    return ei


def _upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float) -> np.ndarray:
    return mu + kappa * sigma


def _kernel_for_dim() -> object:
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-4, 10.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-9, 1.0))


def _fit_gp_with_search(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    dim: int,
) -> Tuple[GaussianProcessRegressor, Dict[str, float]]:
    base_kernel = _kernel_for_dim()
    gp = GaussianProcessRegressor(
        kernel=base_kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )

    param_distributions = {
        "kernel__k1__k1__constant_value": loguniform(1e-3, 1e3),
        "kernel__k1__k2__length_scale": loguniform(1e-4, 10.0),
        "kernel__k2__noise_level": loguniform(1e-9, 1.0),
    }
    n_iter = 6 if dim <= 4 else 4 if dim <= 6 else 3
    cv = 3 if len(y) >= 9 else 2
    search = RandomizedSearchCV(
        gp,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=int(rng.integers(0, 1_000_000)),
        n_jobs=1,
    )
    search.fit(x, y)

    info = {
        "best_score": float(search.best_score_),
        "best_constant": float(search.best_params_["kernel__k1__k1__constant_value"]),
        "best_length_scale": float(search.best_params_["kernel__k1__k2__length_scale"]),
        "best_noise_level": float(search.best_params_["kernel__k2__noise_level"]),
    }
    return search.best_estimator_, info


def _sample_candidates(
    rng: np.random.Generator,
    best_x: np.ndarray,
    n_global: int,
    n_local: int,
    sigma_local: float,
    low: float,
    high: float,
) -> np.ndarray:
    dim = best_x.shape[0]
    global_samples = low + (high - low) * rng.random((n_global, dim))
    local_samples = best_x + rng.normal(0.0, sigma_local, size=(n_local, dim))
    local_samples = np.clip(local_samples, low, high)
    return np.vstack([global_samples, local_samples])


def _apply_boundary_penalty(
    ei: np.ndarray,
    candidates: np.ndarray,
    low: float,
    high: float,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dist_to_low = candidates - low
    dist_to_high = high - candidates
    min_bound_dist = np.minimum(dist_to_low, dist_to_high).min(axis=1)
    # Scale in [0, 1] where 0 is at/over the margin and 1 is safely inside.
    scale = np.clip(min_bound_dist / margin, 0.0, 1.0)
    # Keep a small floor so EI is not fully zeroed near bounds.
    weight = 0.2 + 0.8 * scale
    return ei * weight, min_bound_dist


def _choose_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    boundary_margin: float,
    z_best_threshold: float,
    kappa: float,
    allow_retry: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n_samples, dim = x.shape
    best_idx = int(np.argmax(y))
    best_y = float(y[best_idx])
    best_x = x[best_idx]
    y_median = float(np.median(y))
    y_std = float(np.std(y)) or 1.0
    z_best = (best_y - y_median) / y_std
    acquisition = "ei" if z_best >= z_best_threshold else "ucb"

    gp, search_info = _fit_gp_with_search(x, y, rng, dim)

    n_global = 5000 if dim <= 4 else 7000 if dim <= 6 else 9000
    n_local = 600 if dim <= 4 else 500 if dim <= 6 else 400
    sigma_local = 0.08 if dim <= 4 else 0.10 if dim <= 6 else 0.12

    candidates = _sample_candidates(
        rng=rng,
        best_x=best_x,
        n_global=n_global,
        n_local=n_local,
        sigma_local=sigma_local,
        low=low,
        high=high,
    )

    mu, std = gp.predict(candidates, return_std=True)
    xi = 0.01 * float(np.std(y)) if float(np.std(y)) > 0 else 0.0
    ei = _expected_improvement(mu, std, best_y=best_y, xi=xi)
    ucb = _upper_confidence_bound(mu, std, kappa=kappa)
    if acquisition == "ei":
        scores, bound_dist = _apply_boundary_penalty(ei, candidates, low, high, boundary_margin)
    else:
        scores, bound_dist = _apply_boundary_penalty(ucb, candidates, low, high, boundary_margin)

    # Avoid points that are effectively duplicates of existing samples.
    diff = candidates[:, None, :] - x[None, :, :]
    min_dist = np.min(np.linalg.norm(diff, axis=2), axis=1)
    mask = min_dist > 1e-4
    masked = np.where(mask, scores, -np.inf)
    idx = int(np.argmax(masked))
    if not np.isfinite(masked[idx]):
        idx = int(np.argmax(scores))

    if allow_retry and bound_dist[idx] < boundary_margin:
        tightened_low = low + boundary_margin
        tightened_high = high - boundary_margin
        return _choose_candidate(
            x,
            y,
            rng,
            low=tightened_low,
            high=tightened_high,
            boundary_margin=boundary_margin,
            z_best_threshold=z_best_threshold,
            kappa=kappa,
            allow_retry=False,
        )

    info = {
        "best_y": best_y,
        "y_median": y_median,
        "y_std": y_std,
        "z_best": z_best,
        "acquisition": acquisition,
        "candidate_ei": float(ei[idx]),
        "candidate_ucb": float(ucb[idx]),
        "candidate_min_dist": float(min_dist[idx]),
        "candidate_bound_dist": float(bound_dist[idx]),
    }
    info.update(search_info)
    return candidates[idx], info


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose GP-based candidates for each function.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to data root")
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "submissions"),
        help="Output directory for portal strings",
    )
    parser.add_argument("--seed", type=int, default=20260204, help="Random seed")
    parser.add_argument("--low", type=float, default=0.001, help="Lower bound (inclusive)")
    parser.add_argument("--high", type=float, default=0.98, help="Upper bound (inclusive)")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.05,
        help="Soft margin to discourage extreme boundary inputs",
    )
    parser.add_argument(
        "--z-best-threshold",
        type=float,
        default=2.5,
        help="Use EI when z_best >= threshold, else UCB",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.96,
        help="UCB exploration parameter",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    portal_strings: Dict[str, str] = {}
    raw_vectors: Dict[str, list] = {}
    debug_info: Dict[str, Dict[str, float]] = {}

    for func_id in range(1, 9):
        func_dir = data_root / f"function_{func_id}"
        x = np.load(func_dir / "initial_inputs.npy")
        y = np.load(func_dir / "initial_outputs.npy")
        candidate, info = _choose_candidate(
            x,
            y,
            rng,
            low=args.low,
            high=args.high,
            boundary_margin=args.boundary_margin,
            z_best_threshold=args.z_best_threshold,
            kappa=args.kappa,
        )
        raw_vectors[f"function_{func_id}"] = [float(v) for v in candidate.tolist()]
        portal_strings[f"function_{func_id}"] = "-".join(f"{v:.6f}" for v in candidate)
        debug_info[f"function_{func_id}"] = info

    (out_dir / "round_02_portal_strings.txt").write_text(
        "\n".join(f"{k}: {portal_strings[k]}" for k in sorted(portal_strings.keys())) + "\n",
        encoding="utf-8",
    )
    (out_dir / "round_02_portal_strings.json").write_text(
        json.dumps({"portal_strings": portal_strings, "raw_vectors": raw_vectors}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "round_02_inputs.txt").write_text(
        "[" + ", ".join([repr(np.array(raw_vectors[f"function_{i}"])) for i in range(1, 9)]) + "]\n",
        encoding="utf-8",
    )
    (out_dir / "round_02_gp_debug.json").write_text(
        json.dumps(debug_info, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
