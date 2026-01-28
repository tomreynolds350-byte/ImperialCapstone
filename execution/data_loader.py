from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "initial_data"


def load_function_data(function_id: int, data_root: Path | str = DEFAULT_DATA_ROOT) -> Tuple[np.ndarray, np.ndarray]:
    data_root = Path(data_root)
    func_dir = data_root / f"function_{function_id}"
    x_path = func_dir / "initial_inputs.npy"
    y_path = func_dir / "initial_outputs.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing data for function_{function_id}: {x_path} or {y_path}")
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def iter_functions(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    function_ids: Iterable[int] | None = None,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    ids = list(function_ids) if function_ids is not None else list(range(1, 9))
    for fid in ids:
        x, y = load_function_data(fid, data_root)
        yield fid, x, y
