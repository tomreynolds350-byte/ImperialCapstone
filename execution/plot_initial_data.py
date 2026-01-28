from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import DEFAULT_DATA_ROOT, iter_functions


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (arr - mins) / denom


def plot_histograms(x: np.ndarray, out_path: Path, bins: int = 20) -> None:
    d = x.shape[1]
    ncols = 3
    nrows = int(math.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i in range(d):
        ax = axes[i]
        ax.hist(x[:, i], bins=bins, color="#4c78a8", alpha=0.85)
        ax.set_title(f"x{i + 1}")
    for j in range(d, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_output_distribution(y: np.ndarray, out_path: Path, bins: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y, bins=bins, color="#f58518", alpha=0.85)
    ax.set_title("Output distribution")
    ax.set_xlabel("y")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dim_vs_y(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    d = x.shape[1]
    ncols = 3
    nrows = int(math.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i in range(d):
        ax = axes[i]
        sc = ax.scatter(x[:, i], y, c=y, cmap="viridis", s=40, edgecolor="k", linewidth=0.2)
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel("y")
        ax.set_title(f"x{i + 1} vs y")
    for j in range(d, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_corr_heatmap(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    data = np.column_stack([x, y.reshape(-1, 1)])
    corr = np.corrcoef(data, rowvar=False)
    labels = [f"x{i + 1}" for i in range(x.shape[1])] + ["y"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_2d_scatter(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    if x.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", s=50, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("x1 vs x2 (colored by y)")
    fig.colorbar(sc, ax=ax, label="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_3d_scatter(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    if x.shape[1] < 3:
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap="viridis", s=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("x1, x2, x3 (colored by y)")
    fig.colorbar(sc, ax=ax, pad=0.1, label="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_matrix(x: np.ndarray, y: np.ndarray, out_path: Path, max_dims: int = 5) -> None:
    d = min(x.shape[1], max_dims)
    if d < 2:
        return
    data = x[:, :d]
    fig, axes = plt.subplots(d, d, figsize=(3 * d, 3 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=15, color="#4c78a8", alpha=0.85)
            else:
                ax.scatter(data[:, j], data[:, i], c=y, cmap="viridis", s=20, edgecolor="none")
            if i == d - 1:
                ax.set_xlabel(f"x{j + 1}")
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f"x{i + 1}")
            else:
                ax.set_yticks([])
    fig.suptitle(f"Scatter matrix (first {d} dims)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_parallel_coords(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    d = x.shape[1]
    x_norm = _normalize_01(x)
    y_norm = _normalize_01(y.reshape(-1, 1)).reshape(-1)

    fig, ax = plt.subplots(figsize=(max(6, d * 0.8), 4))
    cmap = plt.cm.viridis
    for i in range(x_norm.shape[0]):
        ax.plot(range(d), x_norm[i], color=cmap(y_norm[i]), alpha=0.6)
    ax.set_xticks(range(d))
    ax.set_xticklabels([f"x{i + 1}" for i in range(d)])
    ax.set_ylabel("normalized value")
    ax.set_title("Parallel coordinates (colored by y)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pca_scores(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def plot_pca_2d(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    if x.shape[1] < 2:
        return
    scores = _pca_scores(x, n_components=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", s=50, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D, colored by y)")
    fig.colorbar(sc, ax=ax, label="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca_3d(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    if x.shape[1] < 3:
        return
    scores = _pca_scores(x, n_components=3)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=y, cmap="viridis", s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA (3D, colored by y)")
    fig.colorbar(sc, ax=ax, pad=0.1, label="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_plots_for_function(x: np.ndarray, y: np.ndarray, out_dir: Path, max_pairplot_dims: int) -> None:
    _ensure_dir(out_dir)
    plot_histograms(x, out_dir / "histograms.png")
    plot_output_distribution(y, out_dir / "output_distribution.png")
    plot_dim_vs_y(x, y, out_dir / "dim_vs_y.png")
    plot_corr_heatmap(x, y, out_dir / "corr_heatmap.png")
    plot_2d_scatter(x, y, out_dir / "scatter_x1_x2.png")
    plot_3d_scatter(x, y, out_dir / "scatter_x1_x2_x3.png")
    plot_scatter_matrix(x, y, out_dir / "scatter_matrix.png", max_dims=max_pairplot_dims)
    plot_parallel_coords(x, y, out_dir / "parallel_coords.png")
    plot_pca_2d(x, y, out_dir / "pca_2d.png")
    plot_pca_3d(x, y, out_dir / "pca_3d.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory plots for initial data.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to initial_data folder")
    parser.add_argument(
        "--out-dir",
        default=str(Path("deliverables") / "initial_data" / "plots"),
        help="Directory for plot outputs",
    )
    parser.add_argument("--max-pairplot-dims", type=int, default=5, help="Max dims for scatter matrix")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    for function_id, x, y in iter_functions(args.data_root):
        func_dir = out_root / f"function_{function_id}"
        generate_plots_for_function(x, y, func_dir, args.max_pairplot_dims)


if __name__ == "__main__":
    main()
