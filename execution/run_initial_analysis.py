from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run summary and plot generation for initial data.")
    parser.add_argument("--data-root", default=str(Path("initial_data")), help="Path to initial_data folder")
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "initial_data"),
        help="Directory for outputs",
    )
    parser.add_argument("--max-pairplot-dims", type=int, default=5, help="Max dims for scatter matrix")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    summary_script = base / "summarize_initial_data.py"
    plot_script = base / "plot_initial_data.py"

    subprocess.check_call(
        [
            sys.executable,
            str(summary_script),
            "--data-root",
            args.data_root,
            "--out-dir",
            args.out_dir,
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            str(plot_script),
            "--data-root",
            args.data_root,
            "--out-dir",
            str(Path(args.out_dir) / "plots"),
            "--max-pairplot-dims",
            str(args.max_pairplot_dims),
        ]
    )


if __name__ == "__main__":
    main()
