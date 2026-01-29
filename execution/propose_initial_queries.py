from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data_loader import DEFAULT_DATA_ROOT, iter_functions


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n + 1)
    samples = np.zeros((n, d))
    for j in range(d):
        points = cut[:n] + (cut[1:] - cut[:n]) * rng.random(n)
        samples[:, j] = points[rng.permutation(n)]
    return samples


def propose_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    num_random: int,
    num_lhs: int,
) -> np.ndarray:
    d = x.shape[1]
    random_candidates = rng.random((num_random, d))
    lhs_candidates = latin_hypercube(num_lhs, d, rng)
    candidates = np.vstack([random_candidates, lhs_candidates])

    dists = np.linalg.norm(candidates[:, None, :] - x[None, :, :], axis=-1)
    min_dist = np.min(dists, axis=1)
    best_idx = int(np.argmax(min_dist))
    candidate = candidates[best_idx]
    return np.clip(candidate, 0.0, 0.999999)


def strategy_params(d: int) -> Tuple[int, int]:
    if d <= 3:
        return 5000, 5000
    if d <= 6:
        return 7000, 5000
    return 10000, 6000


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
        num_random, num_lhs = strategy_params(d)
        candidate = propose_candidate(x, y, rng, num_random, num_lhs)
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
