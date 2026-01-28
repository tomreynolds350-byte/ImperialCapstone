from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data_loader import DEFAULT_DATA_ROOT, iter_functions


def pairwise_median_distance(x: np.ndarray) -> float:
    x = np.asarray(x)
    n = x.shape[0]
    if n < 2:
        return 0.1
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    triu = dists[np.triu_indices(n, k=1)]
    triu = triu[triu > 0]
    if triu.size == 0:
        return 0.1
    return float(np.median(triu))


def propose_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    num_random: int,
    num_local: int,
    local_scale: float,
) -> np.ndarray:
    d = x.shape[1]
    best_idx = int(np.argmax(y))
    best_x = x[best_idx]

    random_candidates = rng.random((num_random, d))
    local_noise = rng.normal(loc=0.0, scale=local_scale, size=(num_local, d))
    local_candidates = np.clip(best_x + local_noise, 0.0, 0.999999)
    candidates = np.vstack([random_candidates, local_candidates])

    lengthscale = pairwise_median_distance(x)
    if lengthscale <= 0:
        lengthscale = 0.1

    dists = np.linalg.norm(candidates[:, None, :] - x[None, :, :], axis=-1)
    weights = np.exp(-(dists ** 2) / (2 * lengthscale ** 2))
    weight_sums = weights.sum(axis=1) + 1e-12
    pred = (weights @ y) / weight_sums

    min_dist = np.min(dists, axis=1)
    y_range = float(np.max(y) - np.min(y))
    explore_weight = 0.1 * y_range if y_range > 0 else 0.1

    scores = pred + explore_weight * min_dist
    best_idx = int(np.argmax(scores))
    candidate = candidates[best_idx]
    return np.clip(candidate, 0.0, 0.999999)


def strategy_params(d: int) -> Tuple[int, int, float]:
    if d <= 3:
        return 6000, 2000, 0.15
    if d <= 6:
        return 5000, 3000, 0.10
    return 4000, 4000, 0.08


def format_portal_string(vec: np.ndarray) -> str:
    return "-".join(f"{v:.6f}" for v in vec.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose initial query points for each function.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to initial_data folder")
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "submissions"),
        help="Directory for submission outputs",
    )
    parser.add_argument("--seed", type=int, default=20260128, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    portal_strings: Dict[str, str] = {}
    raw_vectors: Dict[str, list] = {}

    for function_id, x, y in iter_functions(args.data_root):
        d = x.shape[1]
        num_random, num_local, local_scale = strategy_params(d)
        candidate = propose_candidate(x, y, rng, num_random, num_local, local_scale)
        portal_strings[f"function_{function_id}"] = format_portal_string(candidate)
        raw_vectors[f"function_{function_id}"] = [float(v) for v in candidate.tolist()]

    (out_dir / "round_01_portal_strings.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in portal_strings.items()),
        encoding="utf-8",
    )
    (out_dir / "round_01_portal_strings.json").write_text(
        json.dumps({"portal_strings": portal_strings, "raw_vectors": raw_vectors}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
