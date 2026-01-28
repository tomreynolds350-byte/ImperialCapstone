from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_loader import DEFAULT_DATA_ROOT, iter_functions


def _format_list(values: np.ndarray, precision: int = 6) -> List[float]:
    return [round(float(v), precision) for v in values.tolist()]


def summarize_function(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    summary = {
        "n_samples": int(x.shape[0]),
        "n_dims": int(x.shape[1]) if x.ndim > 1 else 1,
        "input_min": _format_list(np.min(x, axis=0)),
        "input_max": _format_list(np.max(x, axis=0)),
        "input_mean": _format_list(np.mean(x, axis=0)),
        "input_std": _format_list(np.std(x, axis=0, ddof=0)),
        "output_min": float(np.min(y)),
        "output_max": float(np.max(y)),
        "output_mean": float(np.mean(y)),
        "output_std": float(np.std(y, ddof=0)),
        "output_quantiles": {
            "q05": float(np.quantile(y, 0.05)),
            "q25": float(np.quantile(y, 0.25)),
            "q50": float(np.quantile(y, 0.50)),
            "q75": float(np.quantile(y, 0.75)),
            "q95": float(np.quantile(y, 0.95)),
        },
    }
    return summary


def write_json(out_path: Path, data: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(out_path: Path, data: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "function_id",
        "n_samples",
        "n_dims",
        "input_min",
        "input_max",
        "input_mean",
        "input_std",
        "output_min",
        "output_max",
        "output_mean",
        "output_std",
        "output_q05",
        "output_q25",
        "output_q50",
        "output_q75",
        "output_q95",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for function_id, summary in data.items():
            row = {
                "function_id": function_id,
                "n_samples": summary["n_samples"],
                "n_dims": summary["n_dims"],
                "input_min": summary["input_min"],
                "input_max": summary["input_max"],
                "input_mean": summary["input_mean"],
                "input_std": summary["input_std"],
                "output_min": summary["output_min"],
                "output_max": summary["output_max"],
                "output_mean": summary["output_mean"],
                "output_std": summary["output_std"],
                "output_q05": summary["output_quantiles"]["q05"],
                "output_q25": summary["output_quantiles"]["q25"],
                "output_q50": summary["output_quantiles"]["q50"],
                "output_q75": summary["output_quantiles"]["q75"],
                "output_q95": summary["output_quantiles"]["q95"],
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize initial input/output data for all functions.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to initial_data folder")
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "initial_data"),
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary: Dict[str, object] = {}

    for function_id, x, y in iter_functions(args.data_root):
        summary[f"function_{function_id}"] = summarize_function(x, y)

    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "summary.csv", summary)


if __name__ == "__main__":
    main()
