from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from bo_core import (
    append_round_to_initial_data,
    build_round_candidate_parser,
    choose_hybrid_candidate,
    parse_latest_round,
    save_round_outputs_snapshot,
    write_submission_outputs,
)


def _maybe_apply_f5_corner_override(
    func_id: int,
    x: np.ndarray,
    y: np.ndarray,
    candidate: np.ndarray,
    info: Dict[str, Any],
    low: float,
    high: float,
) -> np.ndarray:
    if func_id != 5:
        return candidate

    top_idx = np.argsort(y)[::-1][:2]
    top_points = np.asarray(x[top_idx], dtype=float)
    if top_points.shape[0] < 2:
        return candidate

    if not np.all(top_points >= 0.97):
        return candidate

    if not np.any(np.asarray(candidate, dtype=float) < 0.80):
        return candidate

    override = np.clip(np.mean(top_points, axis=0), low, high)
    info["manual_corner_override_used"] = True
    info["manual_corner_override_reason"] = (
        "Top two observed points occupy the same near-corner basin, "
        "but the model candidate exited that basin on at least one coordinate."
    )
    info["model_candidate_before_override"] = [float(v) for v in np.asarray(candidate, dtype=float).tolist()]
    info["override_candidate"] = [float(v) for v in override.tolist()]
    info["override_anchor_indices"] = [int(i) for i in top_idx.tolist()]
    info["override_anchor_outputs"] = [float(y[i]) for i in top_idx.tolist()]
    info["override_anchor_vectors"] = [[float(v) for v in row.tolist()] for row in top_points]
    return override


def main() -> None:
    parser = build_round_candidate_parser(
        description="Ingest latest round outputs and propose Round 08 candidates (selective exploit hybrid).",
        inputs_default=Path(r"c:\Users\tom_m\Downloads\inputs.txt"),
        outputs_default=Path(r"c:\Users\tom_m\Downloads\outputs.txt"),
        seed_default=20260325,
        prefix_default="round_08",
    )
    parser.set_defaults(strategy="exploit", kappa=1.25, z_best_threshold=1.65, boundary_margin=0.03)
    args = parser.parse_args()

    parsed = parse_latest_round(args.inputs_path, args.outputs_path, args.round_index)
    if args.skip_ingest:
        ingest_summary = {"skipped": True}
    else:
        ingest_summary = append_round_to_initial_data(
            data_root=args.data_root,
            round_inputs=parsed.round_inputs,
            round_outputs=parsed.round_outputs,
        )
        save_round_outputs_snapshot(parsed.round_outputs, args.out_dir, "round_07_outputs_canonical.txt")

    rng = np.random.default_rng(args.seed)
    raw_vectors: Dict[str, List[float]] = {}
    portal_strings: Dict[str, str] = {}
    debug_info: Dict[str, Dict[str, Any]] = {
        "_ingest_summary": ingest_summary,
        "_config": {
            "seed": args.seed,
            "low": args.low,
            "high": args.high,
            "boundary_margin": args.boundary_margin,
            "round_index": args.round_index,
            "inputs_path": str(args.inputs_path),
            "outputs_path": str(args.outputs_path),
            "strategy": args.strategy,
            "kappa": args.kappa,
            "z_best_threshold": args.z_best_threshold,
            "prefix": args.prefix,
            "skip_ingest": bool(args.skip_ingest),
        },
    }

    for func_id in range(1, 9):
        func_key = f"function_{func_id}"
        x = np.load(args.data_root / func_key / "initial_inputs.npy")
        y = np.load(args.data_root / func_key / "initial_outputs.npy").reshape(-1)
        candidate, info = choose_hybrid_candidate(
            x=x,
            y=y,
            rng=rng,
            low=args.low,
            high=args.high,
            boundary_margin=args.boundary_margin,
            seed=args.seed + func_id * 13,
            strategy=args.strategy,
            kappa=args.kappa,
            z_best_threshold=args.z_best_threshold,
        )
        candidate = _maybe_apply_f5_corner_override(
            func_id=func_id,
            x=x,
            y=y,
            candidate=np.asarray(candidate, dtype=float),
            info=info,
            low=args.low,
            high=args.high,
        )
        raw_vectors[func_key] = [float(v) for v in np.asarray(candidate, dtype=float).tolist()]
        portal_strings[func_key] = "-".join(f"{float(v):.6f}" for v in np.asarray(candidate, dtype=float))
        debug_info[func_key] = info

    write_submission_outputs(
        args.out_dir,
        raw_vectors,
        portal_strings,
        debug_info,
        prefix=args.prefix,
        debug_label="hybrid_debug",
    )


if __name__ == "__main__":
    main()
