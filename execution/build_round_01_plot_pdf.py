from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _default_panels() -> List[Tuple[int, str, str]]:
    return [
        (1, "scatter_x1_x2.png", "x1 vs x2 (colored by y)"),
        (1, "dim_vs_y.png", "Dimension vs y"),
        (2, "scatter_x1_x2.png", "x1 vs x2 (colored by y)"),
        (2, "dim_vs_y.png", "Dimension vs y"),
        (3, "scatter_x1_x2.png", "x1 vs x2 (colored by y)"),
        (3, "dim_vs_y.png", "Dimension vs y"),
        (4, "dim_vs_y.png", "Dimension vs y"),
        (4, "parallel_coords.png", "Parallel coordinates"),
        (5, "dim_vs_y.png", "Dimension vs y"),
        (5, "parallel_coords.png", "Parallel coordinates"),
        (6, "dim_vs_y.png", "Dimension vs y"),
        (6, "parallel_coords.png", "Parallel coordinates"),
        (7, "dim_vs_y.png", "Dimension vs y"),
        (7, "parallel_coords.png", "Parallel coordinates"),
        (8, "dim_vs_y.png", "Dimension vs y"),
        (8, "parallel_coords.png", "Parallel coordinates"),
    ]


def _iter_panels(custom: Iterable[str] | None) -> List[Tuple[int, str, str]]:
    if not custom:
        return _default_panels()
    panels: List[Tuple[int, str, str]] = []
    for item in custom:
        parts = item.split("|")
        if len(parts) < 2:
            raise ValueError("Custom panels must be in format function_id|filename|optional_title")
        function_id = int(parts[0])
        filename = parts[1]
        title = parts[2] if len(parts) > 2 else filename
        panels.append((function_id, filename, title))
    return panels


def _load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing plot image: {path}")
    return plt.imread(str(path))


def _plot_side_by_side(
    pre_path: Path,
    post_path: Path,
    title: str,
    function_id: int,
    pdf: PdfPages,
) -> None:
    pre_img = _load_image(pre_path)
    post_img = _load_image(post_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre (initial data)")
    axes[0].axis("off")
    axes[1].imshow(post_img)
    axes[1].set_title("Post (initial + round 01)")
    axes[1].axis("off")
    fig.suptitle(f"Function {function_id} — {title}", fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pre/post round 01 plot PDF.")
    parser.add_argument(
        "--plots-root",
        default=str(Path("deliverables") / "round_01" / "plots"),
        help="Root folder containing pre/ and post/ plot folders",
    )
    parser.add_argument(
        "--out-path",
        default=str(Path("deliverables") / "round_01" / "plots" / "round_01_pre_post_summary.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--panel",
        action="append",
        help="Custom panel in format function_id|filename|optional_title (can repeat)",
    )
    args = parser.parse_args()

    plots_root = Path(args.plots_root)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels = _iter_panels(args.panel)

    with PdfPages(out_path) as pdf:
        for function_id, filename, title in panels:
            pre_path = plots_root / "pre" / f"function_{function_id}" / filename
            post_path = plots_root / "post" / f"function_{function_id}" / filename
            _plot_side_by_side(pre_path, post_path, title, function_id, pdf)


if __name__ == "__main__":
    main()
