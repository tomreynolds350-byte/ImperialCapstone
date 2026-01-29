from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from data_loader import DEFAULT_DATA_ROOT, iter_functions


def pca_explained_variance_ratio(x: np.ndarray, n_components: int = 3) -> List[float]:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    var = s ** 2
    total = np.sum(var)
    if total == 0:
        return [0.0] * n_components
    ratios = (var / total)[:n_components]
    return [float(r) for r in ratios]


def corr_with_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    data = np.column_stack([x, y.reshape(-1, 1)])
    corr = np.corrcoef(data, rowvar=False)
    return corr[:-1, -1]


def format_range(values: np.ndarray) -> str:
    return f"{float(np.min(values)):.4f} to {float(np.max(values)):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an internal plot guide for initial data.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to initial_data folder")
    parser.add_argument(
        "--out-path",
        default=str(Path("deliverables") / "notes" / "round_01_plot_guide.md"),
        help="Output markdown path",
    )
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Round 01 Plot Guide (Internal)")
    lines.append("")
    lines.append("This is an internal, novice-friendly guide to explain what the PCA and scatter plots show for each function.")
    lines.append("It stays descriptive: ranges, variance structure, and which dimensions are most associated with y in the current data.")
    lines.append("")
    lines.append("## How to read the plots (quick primer)")
    lines.append("")
    lines.append("- Scatter x1 vs x2 (colored by y): shows coverage in 2D; color gradients indicate where y is higher or lower.")
    lines.append("- Dim vs y scatter: one plot per dimension; a visible slope suggests that dimension may influence y.")
    lines.append("- PCA (2D/3D): compresses inputs into principal components; the % variance tells you how much of the input spread is captured.")
    lines.append("- Correlation heatmap: linear association between each x and y (positive/negative).")
    lines.append("- Parallel coordinates: all dimensions on one chart; color highlights whether higher y aligns with certain ranges.")
    lines.append("")

    for function_id, x, y in iter_functions(args.data_root):
        y = y.reshape(-1)
        d = x.shape[1]
        ratios = pca_explained_variance_ratio(x, n_components=min(3, d))
        corr = corr_with_y(x, y)
        abs_corr = np.abs(corr)
        top_idx = int(np.argmax(abs_corr))

        lines.append(f"## Function {function_id}")
        lines.append("")
        lines.append(f"- Samples: {x.shape[0]}, Dimensions: {d}")
        lines.append(f"- y range: {format_range(y)} (mean {float(np.mean(y)):.4f}, std {float(np.std(y)):.4f})")
        lines.append(f"- x ranges: " + ", ".join([f"x{i+1} {format_range(x[:, i])}" for i in range(d)]))

        if d >= 2:
            lines.append(
                f"- PCA variance (PC1/PC2{'/PC3' if d >= 3 else ''}): "
                + ", ".join([f"{r*100:.1f}%" for r in ratios])
            )
        else:
            lines.append("- PCA variance: not applicable (1D)")

        lines.append(
            f"- Strongest linear association with y (by |corr|): x{top_idx + 1} (corr {corr[top_idx]:.3f})"
        )
        lines.append(
            "- Plots to review: "
            f"plots/function_{function_id}/scatter_x1_x2.png, "
            f"plots/function_{function_id}/dim_vs_y.png, "
            f"plots/function_{function_id}/pca_2d.png"
        )
        lines.append("")
        lines.append("Suggested talking points:")
        lines.append(
            "- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension."
        )
        lines.append(
            "- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread."
        )
        lines.append(
            "- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape."
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
