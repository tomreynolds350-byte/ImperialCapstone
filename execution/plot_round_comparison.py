from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from data_loader import DEFAULT_DATA_ROOT, load_function_data
from plot_initial_data import generate_plots_for_function


def _safe_eval_list(path: Path, extra_globals: dict) -> List[object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty round data file: {path}")
    scope = {"__builtins__": {}}
    scope.update(extra_globals)
    data = eval(text, scope)  # noqa: S307 - controlled globals for local artifact parsing
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected list/tuple in {path}, got {type(data)}")
    return list(data)


def load_round_inputs(path: Path) -> List[np.ndarray]:
    raw = _safe_eval_list(path, {"np": np, "array": np.array})
    return [np.asarray(item, dtype=float).reshape(-1) for item in raw]


def load_round_outputs(path: Path) -> List[float]:
    raw = _safe_eval_list(path, {"np": np})
    return [float(item) for item in raw]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre/post round plots using existing plotting functions."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to initial_data folder")
    parser.add_argument(
        "--round-inputs",
        default=str(Path("deliverables") / "submissions" / "round_01_inputs.txt"),
        help="Path to round inputs text file",
    )
    parser.add_argument(
        "--round-outputs",
        default=str(Path("deliverables") / "submissions" / "round_01_outputs.txt"),
        help="Path to round outputs text file",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "round_01" / "plots"),
        help="Output directory for plot images",
    )
    parser.add_argument("--max-pairplot-dims", type=int, default=5, help="Max dims for scatter matrix")
    args = parser.parse_args()

    round_inputs = load_round_inputs(Path(args.round_inputs))
    round_outputs = load_round_outputs(Path(args.round_outputs))
    if len(round_inputs) != 8 or len(round_outputs) != 8:
        raise ValueError("Expected 8 inputs and 8 outputs for round 01.")

    out_root = Path(args.out_dir)
    for function_id in range(1, 9):
        pre_x, pre_y = load_function_data(function_id, args.data_root)
        r_x = round_inputs[function_id - 1]
        r_y = round_outputs[function_id - 1]

        if pre_x.shape[1] != r_x.shape[0]:
            raise ValueError(
                f"Dimension mismatch for function_{function_id}: "
                f"initial has {pre_x.shape[1]} dims, round input has {r_x.shape[0]} dims"
            )

        post_x = np.vstack([pre_x, r_x.reshape(1, -1)])
        post_y = np.concatenate([pre_y.reshape(-1), [r_y]])

        generate_plots_for_function(pre_x, pre_y, out_root / "pre" / f"function_{function_id}", args.max_pairplot_dims)
        generate_plots_for_function(
            post_x, post_y, out_root / "post" / f"function_{function_id}", args.max_pairplot_dims
        )


if __name__ == "__main__":
    main()
