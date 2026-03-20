from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "initial_data"
DEFAULT_OUT_DIR = REPO_ROOT / "deliverables" / "submissions"

DEFAULT_LOW = 0.001
DEFAULT_HIGH = 0.98
LENGTH_SCALE_BOUNDS = (0.02, 3.0)
NOISE_LEVEL_BOUNDS = (1e-6, 0.25)
SHORTLIST_SIZE_BY_STRATEGY = {"balanced": 96, "explore": 128, "exploit": 72}
HYBRID_WEIGHTS = {
    "balanced": {"gp": 0.70, "nn": 0.15, "classification": 0.10, "novelty": 0.05},
    "explore": {"gp": 0.55, "nn": 0.10, "classification": 0.10, "novelty": 0.25},
    "exploit": {"gp": 0.76, "nn": 0.14, "classification": 0.08, "novelty": 0.02},
}


@dataclass
class ParsedRoundData:
    round_inputs: List[np.ndarray]
    round_outputs: List[float]


@dataclass
class ScoreStats:
    gp_mean: float
    gp_std: float
    nn_mean: float
    nn_std: float
    cls_mean: float
    cls_std: float
    novelty_mean: float
    novelty_std: float


class _WorkspaceTemporaryDirectory:
    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
    ) -> None:
        self._ignore_cleanup_errors = ignore_cleanup_errors
        suffix = suffix or ""
        prefix = prefix or "tmp"
        base_dir = Path(dir) if dir is not None else (REPO_ROOT / ".tmp")
        _ensure_writable_dir(base_dir)
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        candidate.mkdir(parents=True, exist_ok=False)
        self.name = str(candidate)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


def _target_scale(y: np.ndarray) -> float:
    return max(float(np.std(y)), 1e-12)


def _safe_score_value(value: float, clip_min: float = -10.0, clip_max: float = 1.0) -> float:
    if not np.isfinite(value):
        return clip_min
    return float(np.clip(value, clip_min, clip_max))


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.std(y_true)) < 1e-6:
        return 0.0
    return _safe_score_value(float(r2_score(y_true, y_pred)))


def _mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)) + 1e-9


def _standardize(values: np.ndarray) -> np.ndarray:
    mean, std = _mean_and_std(values)
    return (np.asarray(values, dtype=float) - mean) / std


def _format_portal_line(function_id: int, vector: np.ndarray) -> str:
    return f"function_{function_id}: " + "-".join(f"{float(v):.6f}" for v in vector)


def _ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _native_tempdir_is_unwritable() -> bool:
    probe_root = REPO_ROOT / ".tmp"
    _ensure_writable_dir(probe_root)
    try:
        with tempfile.TemporaryDirectory(dir=str(probe_root)) as tmp_dir:
            probe_file = Path(tmp_dir) / ".write_test.txt"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
        return False
    except Exception:
        return True


if _native_tempdir_is_unwritable():
    tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory


def _extract_top_level_lists(text: str) -> List[str]:
    chunks: List[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start : i + 1])
                start = None
    if depth != 0:
        raise ValueError("Unbalanced brackets while parsing batch file.")
    return chunks


def parse_batch_file(path: Path) -> List[List[Any]]:
    text = path.read_text(encoding="utf-8")
    chunks = _extract_top_level_lists(text)
    parsed: List[List[Any]] = []
    eval_ctx = {"np": np, "array": np.array}
    for chunk in chunks:
        parsed.append(eval(chunk, eval_ctx, {}))  # noqa: S307 - trusted local artifact
    return parsed


def parse_latest_round(inputs_path: Path, outputs_path: Path, round_index: int) -> ParsedRoundData:
    input_batches = parse_batch_file(inputs_path)
    output_batches = parse_batch_file(outputs_path)
    if not input_batches or not output_batches:
        raise ValueError("No round batches found in inputs/outputs files.")
    if len(input_batches) != len(output_batches):
        raise ValueError(
            f"Batch count mismatch: inputs={len(input_batches)} outputs={len(output_batches)}"
        )

    idx = round_index
    if idx < 0:
        idx = len(input_batches) + idx
    if idx < 0 or idx >= len(input_batches):
        raise ValueError(f"Invalid round index {round_index}; available batches: {len(input_batches)}")

    raw_inputs = input_batches[idx]
    raw_outputs = output_batches[idx]
    if len(raw_inputs) != 8 or len(raw_outputs) != 8:
        raise ValueError("Each round batch must contain exactly 8 functions.")

    round_inputs = [np.asarray(v, dtype=float).reshape(-1) for v in raw_inputs]
    round_outputs = [float(v) for v in raw_outputs]
    return ParsedRoundData(round_inputs=round_inputs, round_outputs=round_outputs)


def save_round_outputs_snapshot(round_outputs: Sequence[float], out_dir: Path, filename: str) -> None:
    _ensure_writable_dir(out_dir)
    values = ", ".join(f"np.float64({float(v)!r})" for v in round_outputs)
    (out_dir / filename).write_text(f"[{values}]\n", encoding="utf-8")


def append_round_to_initial_data(
    data_root: Path,
    round_inputs: Sequence[np.ndarray],
    round_outputs: Sequence[float],
    atol: float = 1e-10,
) -> Dict[str, Dict[str, Any]]:
    ingest_summary: Dict[str, Dict[str, Any]] = {}

    for func_id in range(1, 9):
        func_key = f"function_{func_id}"
        func_dir = data_root / func_key
        x_path = func_dir / "initial_inputs.npy"
        y_path = func_dir / "initial_outputs.npy"
        x = np.load(x_path)
        y = np.load(y_path).reshape(-1)

        x_new = np.asarray(round_inputs[func_id - 1], dtype=float).reshape(1, -1)
        y_new = float(round_outputs[func_id - 1])

        if x.shape[1] != x_new.shape[1]:
            raise ValueError(
                f"Dimension mismatch for {func_key}: existing={x.shape[1]} new={x_new.shape[1]}"
            )

        existing_mask = np.all(np.isclose(x, x_new, atol=atol, rtol=0.0), axis=1)
        appended = False
        duplicate_index = None
        if np.any(existing_mask):
            duplicate_index = int(np.flatnonzero(existing_mask)[0])
            y_existing = float(y[duplicate_index])
            if abs(y_existing - y_new) > 1e-8:
                raise ValueError(
                    f"{func_key} duplicate input has mismatched output: existing={y_existing} new={y_new}"
                )
        else:
            x = np.vstack([x, x_new])
            y = np.concatenate([y, np.array([y_new], dtype=float)])
            np.save(x_path, x)
            np.save(y_path, y)
            appended = True

        ingest_summary[func_key] = {
            "appended": appended,
            "duplicate_index": duplicate_index,
            "n_samples_after": int(x.shape[0]),
            "new_output": y_new,
        }

    return ingest_summary


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    improvement = np.asarray(mu, dtype=float) - float(best_y) - float(xi)
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-9] = 0.0
    return ei


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float) -> np.ndarray:
    return np.asarray(mu, dtype=float) + float(kappa) * np.asarray(sigma, dtype=float)


def _kernel_for_dim(dim: int) -> object:
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.full(dim, 0.2, dtype=float),
        length_scale_bounds=LENGTH_SCALE_BOUNDS,
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=NOISE_LEVEL_BOUNDS)


def _fit_gp_with_fixed_kernel(
    x: np.ndarray,
    y: np.ndarray,
    kernel: object,
) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=clone(kernel),
        normalize_y=True,
        optimizer=None,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(x, y)
    return gp


def _loo_gp_metrics(x: np.ndarray, y: np.ndarray, fitted_gp: GaussianProcessRegressor) -> Tuple[float, float]:
    if len(y) < 2:
        return 0.0, 0.0

    preds = np.zeros(len(y), dtype=float)
    for idx in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        loo_gp = _fit_gp_with_fixed_kernel(x[mask], y[mask], fitted_gp.kernel_)
        preds[idx] = float(loo_gp.predict(x[idx].reshape(1, -1))[0])
    loo_mae = float(mean_absolute_error(y, preds))
    loo_rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return loo_mae, loo_rmse


def fit_gp_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    n_restarts_optimizer: int = 8,
) -> Tuple[GaussianProcessRegressor, Dict[str, Any]]:
    dim = x.shape[1]
    gp = GaussianProcessRegressor(
        kernel=_kernel_for_dim(dim),
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    fit_used_fallback = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            gp.fit(x, y)
        except Exception:
            fit_used_fallback = True
            gp = _fit_gp_with_fixed_kernel(x, y, _kernel_for_dim(dim))

    loo_mae, loo_rmse = _loo_gp_metrics(x, y, gp)

    fitted_length_scales = np.asarray(gp.kernel_.k1.k2.length_scale, dtype=float).reshape(-1)
    lower_hits = np.isclose(fitted_length_scales, LENGTH_SCALE_BOUNDS[0], rtol=0.0, atol=1e-4)
    upper_hits = np.isclose(fitted_length_scales, LENGTH_SCALE_BOUNDS[1], rtol=0.0, atol=1e-4)

    info: Dict[str, Any] = {
        "fit_score_name": "negative_loo_rmse",
        "best_score": float(-loo_rmse),
        "best_constant": float(gp.kernel_.k1.k1.constant_value),
        "best_length_scale": float(np.median(fitted_length_scales)),
        "best_length_scales": [float(v) for v in fitted_length_scales.tolist()],
        "best_noise_level": float(gp.kernel_.k2.noise_level),
        "objective_direction": "maximize",
        "target_std": float(np.std(y)),
        "loo_mae": loo_mae,
        "loo_rmse": loo_rmse,
        "length_scale_at_lower_bound": bool(np.any(lower_hits)),
        "length_scale_at_upper_bound": bool(np.any(upper_hits)),
        "length_scale_lower_bound_dims_1_based": [int(i + 1) for i in np.flatnonzero(lower_hits)],
        "length_scale_upper_bound_dims_1_based": [int(i + 1) for i in np.flatnonzero(upper_hits)],
        "n_optimizer_restarts": int(n_restarts_optimizer),
        "gp_fit_fallback": bool(fit_used_fallback),
    }
    return gp, info


def _fit_mlp_regressor(x: np.ndarray, y: np.ndarray, seed: int) -> TransformedTargetRegressor:
    dim = x.shape[1]
    hidden = (max(8, dim * 4), max(4, dim * 2))
    mlp = Pipeline(
        steps=[
            ("x_scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1e-3,
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model = TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x, y)
    return model


def _fit_classifiers(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[Any, Any, np.ndarray, float]:
    threshold = float(np.quantile(y, 0.70))
    labels = (y >= threshold).astype(int)
    if labels.min() == labels.max():
        threshold = float(np.quantile(y, 0.50))
        labels = (y >= threshold).astype(int)
    if labels.min() == labels.max():
        labels = np.zeros_like(y, dtype=int)
        labels[int(np.argmax(y))] = 1

    logistic = Pipeline(
        steps=[
            ("x_scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )

    svc = Pipeline(
        steps=[
            ("x_scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=3.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    logistic.fit(x, labels)
    svc.fit(x, labels)
    return logistic, svc, labels, threshold


def _evaluate_regression_models(x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
    n = len(y)
    if n < 4:
        return {
            "linear_r2_cv_mean": 0.0,
            "svr_r2_cv_mean": 0.0,
            "mlp_r2_cv_mean": 0.0,
        }

    dim = x.shape[1]
    n_splits = 3 if n >= 9 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    linear = Pipeline([("x_scale", StandardScaler()), ("model", LinearRegression())])
    svr = Pipeline([("x_scale", StandardScaler()), ("model", SVR(C=10.0, gamma="scale"))])

    scores_linear: List[float] = []
    scores_svr: List[float] = []
    scores_mlp: List[float] = []

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        linear.fit(x_train, y_train)
        scores_linear.append(_safe_r2_score(y_test, linear.predict(x_test)))

        svr.fit(x_train, y_train)
        scores_svr.append(_safe_r2_score(y_test, svr.predict(x_test)))

        mlp = _fit_mlp_regressor(x_train, y_train, seed=seed + dim + len(train_idx))
        scores_mlp.append(_safe_r2_score(y_test, mlp.predict(x_test)))

    return {
        "linear_r2_cv_mean": float(np.mean(scores_linear)),
        "svr_r2_cv_mean": float(np.mean(scores_svr)),
        "mlp_r2_cv_mean": float(np.mean(scores_mlp)),
    }


def reflect_to_bounds(values: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    span = float(high) - float(low)
    if span <= 0:
        raise ValueError("high must be greater than low")
    shifted = np.mod(arr - low, 2.0 * span)
    reflected = np.where(shifted <= span, shifted, 2.0 * span - shifted)
    return reflected + low


def _sample_local_cloud(
    rng: np.random.Generator,
    center: np.ndarray,
    n_samples: int,
    sigma: float,
    low: float,
    high: float,
) -> np.ndarray:
    raw = center + rng.normal(0.0, sigma, size=(n_samples, center.shape[0]))
    return reflect_to_bounds(raw, low, high)


def _boundary_distance(points: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    return np.minimum(arr - low, high - arr).min(axis=1)


def _boundary_weight(bound_dist: np.ndarray, boundary_margin: float, floor: float = 0.25) -> np.ndarray:
    margin = max(float(boundary_margin), 1e-9)
    scale = np.clip(np.asarray(bound_dist, dtype=float) / margin, 0.0, 1.0)
    return floor + (1.0 - floor) * scale


def _min_distance_to_dataset(candidates: np.ndarray, x: np.ndarray) -> np.ndarray:
    diff = np.asarray(candidates, dtype=float)[:, None, :] - np.asarray(x, dtype=float)[None, :, :]
    return np.min(np.linalg.norm(diff, axis=2), axis=1)


def _gp_candidate_pool(
    rng: np.random.Generator,
    best_x: np.ndarray,
    dim: int,
    low: float,
    high: float,
) -> np.ndarray:
    n_global = 5000 if dim <= 4 else 7000 if dim <= 6 else 9000
    n_local = 600 if dim <= 4 else 500 if dim <= 6 else 400
    sigma_local = 0.08 if dim <= 4 else 0.10 if dim <= 6 else 0.12

    global_samples = low + (high - low) * rng.random((n_global, dim))
    local_samples = _sample_local_cloud(rng, best_x, n_local, sigma_local, low, high)
    return np.vstack([global_samples, local_samples])


def _hybrid_candidate_pool(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    support_indices: np.ndarray,
    low: float,
    high: float,
    strategy: str,
) -> np.ndarray:
    dim = x.shape[1]
    top_k = min(4, len(y))
    top_idx = np.argsort(y)[-top_k:]

    if strategy == "explore":
        n_global = 18000 if dim <= 4 else 22000 if dim <= 6 else 26000
        n_local_per_top = 120 if dim <= 4 else 90 if dim <= 6 else 70
        n_support = 360 if support_indices.size > 0 else 0
        sigma_local = 0.10 if dim <= 4 else 0.12 if dim <= 6 else 0.14
    elif strategy == "exploit":
        n_global = 4500 if dim <= 4 else 6000 if dim <= 6 else 7500
        n_local_per_top = 650 if dim <= 4 else 500 if dim <= 6 else 350
        n_support = 140 if support_indices.size > 0 else 0
        sigma_local = 0.045 if dim <= 4 else 0.06 if dim <= 6 else 0.08
    else:
        n_global = 8000 if dim <= 4 else 10000 if dim <= 6 else 12000
        n_local_per_top = 450 if dim <= 4 else 350 if dim <= 6 else 250
        n_support = 200 if support_indices.size > 0 else 0
        sigma_local = 0.06 if dim <= 4 else 0.08 if dim <= 6 else 0.10

    global_samples = low + (high - low) * rng.random((n_global, dim))
    parts: List[np.ndarray] = [global_samples]

    for idx in top_idx:
        parts.append(_sample_local_cloud(rng, x[int(idx)], n_local_per_top, sigma_local, low, high))

    for idx in support_indices[:3]:
        parts.append(_sample_local_cloud(rng, x[int(idx)], n_support, sigma_local * 0.75, low, high))

    parts.append(np.asarray(x[top_idx], dtype=float))
    return np.vstack(parts)


def _choose_acquisition(strategy: str, z_best: float, z_best_threshold: float) -> str:
    if strategy == "explore":
        return "ucb"
    return "ei" if z_best >= z_best_threshold else "ucb"


def _hybrid_score_components(
    gp_values: np.ndarray,
    nn_values: np.ndarray,
    cls_values: np.ndarray,
    novelty_values: np.ndarray,
) -> ScoreStats:
    gp_mean, gp_std = _mean_and_std(gp_values)
    nn_mean, nn_std = _mean_and_std(nn_values)
    cls_mean, cls_std = _mean_and_std(cls_values)
    novelty_mean, novelty_std = _mean_and_std(novelty_values)
    return ScoreStats(
        gp_mean=gp_mean,
        gp_std=gp_std,
        nn_mean=nn_mean,
        nn_std=nn_std,
        cls_mean=cls_mean,
        cls_std=cls_std,
        novelty_mean=novelty_mean,
        novelty_std=novelty_std,
    )


def _hybrid_total_score(
    *,
    gp_value: float,
    nn_value: float,
    cls_value: float,
    novelty_value: float,
    boundary_value: float,
    stats: ScoreStats,
    weights: Dict[str, float],
) -> float:
    gp_norm = (float(gp_value) - stats.gp_mean) / stats.gp_std
    nn_norm = (float(nn_value) - stats.nn_mean) / stats.nn_std
    cls_norm = (float(cls_value) - stats.cls_mean) / stats.cls_std
    nov_norm = (float(novelty_value) - stats.novelty_mean) / stats.novelty_std
    weighted = (
        weights["gp"] * gp_norm
        + weights["nn"] * nn_norm
        + weights["classification"] * cls_norm
        + weights["novelty"] * nov_norm
    )
    return float(weighted * boundary_value)


def _refine_candidates(
    rng: np.random.Generator,
    x: np.ndarray,
    start_points: np.ndarray,
    score_fn: Any,
    low: float,
    high: float,
    *,
    strategy: str,
    novelty_floor: float,
    duplicate_tol: float = 1e-5,
) -> List[Tuple[np.ndarray, float]]:
    dim = x.shape[1]
    max_points = min(15, len(start_points))
    initial_step = 0.04 if dim <= 4 else 0.03 if dim <= 6 else 0.025
    if strategy == "explore":
        initial_step *= 0.9
    elif strategy == "exploit":
        initial_step *= 0.75

    results: List[Tuple[np.ndarray, float]] = []
    for point in np.asarray(start_points[:max_points], dtype=float):
        current = point.copy()
        current_score = float(score_fn(current))
        results.append((current.copy(), current_score))
        step = initial_step

        for _ in range(8):
            proposals = [current]
            for _ in range(6):
                delta = rng.normal(0.0, step, size=dim)
                trial = reflect_to_bounds(current + delta, low, high)
                if float(np.min(np.linalg.norm(x - trial.reshape(1, -1), axis=1))) <= duplicate_tol:
                    continue
                if strategy == "explore":
                    trial_min_dist = float(np.min(np.linalg.norm(x - trial.reshape(1, -1), axis=1)))
                    if trial_min_dist < novelty_floor:
                        continue
                proposals.append(trial)

            proposal_scores = [float(score_fn(p)) for p in proposals]
            best_idx = int(np.argmax(proposal_scores))
            best_point = proposals[best_idx]
            best_score = proposal_scores[best_idx]

            if best_score > current_score:
                current = np.asarray(best_point, dtype=float)
                current_score = float(best_score)
                results.append((current.copy(), current_score))
            else:
                step *= 0.5
                if step < 0.004:
                    break

    return results


def _boundary_override(
    candidates_with_scores: List[Tuple[np.ndarray, float]],
    low: float,
    high: float,
    boundary_margin: float,
) -> Tuple[np.ndarray, float, bool]:
    if not candidates_with_scores:
        raise ValueError("No candidates available for boundary override")

    scored = [
        (
            np.asarray(point, dtype=float),
            float(score),
            float(_boundary_distance(np.asarray(point, dtype=float).reshape(1, -1), low, high)[0]),
        )
        for point, score in candidates_with_scores
    ]
    scored.sort(key=lambda item: item[1])
    best_point, best_score, best_bound_dist = scored[-1]

    interior = [item for item in scored if item[2] >= boundary_margin]
    if best_bound_dist >= boundary_margin or not interior:
        return best_point, best_score, False

    interior_point, interior_score, _ = interior[-1]
    required_score = interior_score + 0.05 * max(abs(interior_score), 1.0)
    if best_score >= required_score:
        return best_point, best_score, False
    return interior_point, interior_score, True


def choose_gp_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    low: float,
    high: float,
    boundary_margin: float,
    z_best_threshold: float,
    kappa: float,
    gp_restarts: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples, dim = x.shape
    best_idx = int(np.argmax(y))
    best_y = float(y[best_idx])
    best_x = x[best_idx]
    y_median = float(np.median(y))
    y_std = _target_scale(y)
    z_best = (best_y - y_median) / y_std
    acquisition = _choose_acquisition("balanced", z_best, z_best_threshold)

    gp, gp_info = fit_gp_model(
        x,
        y,
        random_state=int(rng.integers(0, 1_000_000)),
        n_restarts_optimizer=gp_restarts,
    )

    candidates = _gp_candidate_pool(rng, best_x, dim, low, high)
    min_dist = _min_distance_to_dataset(candidates, x)
    bound_dist = _boundary_distance(candidates, low, high)
    boundary_weight = _boundary_weight(bound_dist, boundary_margin, floor=0.20)

    mu, sigma = gp.predict(candidates, return_std=True)
    xi = 0.01 * float(np.std(y)) if float(np.std(y)) > 0 else 0.0
    ei = expected_improvement(mu, sigma, best_y=best_y, xi=xi)
    ucb = upper_confidence_bound(mu, sigma, kappa=kappa)
    gp_primary = ei if acquisition == "ei" else ucb

    valid_mask = min_dist > 1e-5
    shortlist_score = gp_primary * boundary_weight
    valid_scores = np.where(valid_mask, shortlist_score, -np.inf)
    shortlist_size = min(64, len(candidates))
    shortlist_idx = np.argsort(valid_scores)[-shortlist_size:]
    shortlist_idx = shortlist_idx[np.isfinite(valid_scores[shortlist_idx])]
    if shortlist_idx.size == 0:
        shortlist_idx = np.array([int(np.argmax(shortlist_score))], dtype=int)

    start_points = candidates[shortlist_idx[::-1]]

    def score_fn(point: np.ndarray) -> float:
        point_2d = np.asarray(point, dtype=float).reshape(1, -1)
        mu_s, sigma_s = gp.predict(point_2d, return_std=True)
        ei_s = expected_improvement(mu_s, sigma_s, best_y=best_y, xi=xi)
        ucb_s = upper_confidence_bound(mu_s, sigma_s, kappa=kappa)
        gp_s = float(ei_s[0] if acquisition == "ei" else ucb_s[0])
        bound_s = float(_boundary_distance(point_2d, low, high)[0])
        return float(gp_s * _boundary_weight(np.array([bound_s]), boundary_margin, floor=0.20)[0])

    refined = _refine_candidates(
        rng,
        x,
        start_points,
        score_fn,
        low,
        high,
        strategy="balanced",
        novelty_floor=0.0,
    )
    for point in start_points:
        refined.append((np.asarray(point, dtype=float), float(score_fn(point))))

    chosen, chosen_score, boundary_override_used = _boundary_override(refined, low, high, boundary_margin)
    chosen_min_dist = float(np.min(np.linalg.norm(x - chosen.reshape(1, -1), axis=1)))
    chosen_bound_dist = float(_boundary_distance(chosen.reshape(1, -1), low, high)[0])
    chosen_mu, chosen_sigma = gp.predict(chosen.reshape(1, -1), return_std=True)
    chosen_ei = expected_improvement(chosen_mu, chosen_sigma, best_y=best_y, xi=xi)
    chosen_ucb = upper_confidence_bound(chosen_mu, chosen_sigma, kappa=kappa)

    info: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_dims": int(dim),
        "best_y": best_y,
        "best_index": best_idx,
        "y_median": y_median,
        "y_std": float(np.std(y)),
        "z_best": float(z_best),
        "acquisition": acquisition,
        "candidate_ei": float(chosen_ei[0]),
        "candidate_ucb": float(chosen_ucb[0]),
        "candidate_min_dist": chosen_min_dist,
        "candidate_bound_dist": chosen_bound_dist,
        "chosen_candidate_score": float(chosen_score),
        "chosen_candidate_min_dist": chosen_min_dist,
        "chosen_candidate_bound_dist": chosen_bound_dist,
        "boundary_override_used": bool(boundary_override_used),
        "ucb_kappa": float(kappa),
    }
    info.update(gp_info)
    return chosen, info


def choose_hybrid_candidate(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    low: float,
    high: float,
    boundary_margin: float,
    seed: int,
    strategy: str,
    kappa: float,
    z_best_threshold: float = 2.2,
    gp_restarts: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples, dim = x.shape
    y_std = _target_scale(y)
    y_median = float(np.median(y))
    best_idx = int(np.argmax(y))
    best_y = float(y[best_idx])
    z_best = (best_y - y_median) / y_std
    acquisition = _choose_acquisition(strategy, z_best, z_best_threshold)

    gp, gp_info = fit_gp_model(
        x,
        y,
        random_state=seed + 17,
        n_restarts_optimizer=gp_restarts,
    )
    mlp = _fit_mlp_regressor(x, y, seed=seed + dim)
    logistic, svc, labels, cls_threshold = _fit_classifiers(x, y, seed=seed + 100 + dim)
    regression_metrics = _evaluate_regression_models(x, y, seed=seed + 200 + dim)

    svc_model = svc.named_steps["svc"]
    train_decision = np.asarray(svc.decision_function(x), dtype=float)
    support_indices = np.asarray(svc_model.support_, dtype=int)
    support_set = set(support_indices.tolist())
    boundary_order = np.argsort(np.abs(train_decision))
    support_boundary_sorted = [int(i) for i in boundary_order if int(i) in support_set]
    if not support_boundary_sorted:
        support_boundary_sorted = [int(i) for i in support_indices.tolist()]
    support_for_sampling = np.asarray(support_boundary_sorted[:3], dtype=int)

    candidates = _hybrid_candidate_pool(
        rng=rng,
        x=x,
        y=y,
        support_indices=support_for_sampling,
        low=low,
        high=high,
        strategy=strategy,
    )

    mu, sigma = gp.predict(candidates, return_std=True)
    xi = 0.01 * float(np.std(y)) if float(np.std(y)) > 0 else 0.0
    ei = expected_improvement(mu, sigma, best_y=best_y, xi=xi)
    ucb = upper_confidence_bound(mu, sigma, kappa=kappa)
    gp_primary = ei if acquisition == "ei" else ucb

    nn_pred = np.asarray(mlp.predict(candidates), dtype=float)
    p_good_log = logistic.predict_proba(candidates)[:, 1]
    p_good_svc = svc.predict_proba(candidates)[:, 1]
    p_good = 0.5 * (p_good_log + p_good_svc)
    p_boundary = 1.0 - np.abs(p_good - 0.5) * 2.0
    cls_mix = 0.75 * p_good + 0.25 * p_boundary

    min_dist = _min_distance_to_dataset(candidates, x)
    min_bound_dist = _boundary_distance(candidates, low, high)
    boundary_weight = _boundary_weight(min_bound_dist, boundary_margin, floor=0.25)

    valid_mask = min_dist > 1e-5
    novelty_floor = 0.0
    if strategy == "explore":
        novelty_floor = float(np.quantile(min_dist, 0.65))
        valid_mask = valid_mask & (min_dist >= novelty_floor)
        if not np.any(valid_mask):
            novelty_floor = float(np.quantile(min_dist, 0.50))
            valid_mask = min_dist >= novelty_floor

    gp_shortlist_score = gp_primary * boundary_weight
    valid_gp_scores = np.where(valid_mask, gp_shortlist_score, -np.inf)
    shortlist_size = min(SHORTLIST_SIZE_BY_STRATEGY[strategy], len(candidates))
    shortlist_idx = np.argsort(valid_gp_scores)[-shortlist_size:]
    shortlist_idx = shortlist_idx[np.isfinite(valid_gp_scores[shortlist_idx])]
    if shortlist_idx.size == 0:
        shortlist_idx = np.array([int(np.argmax(gp_shortlist_score))], dtype=int)

    short_candidates = candidates[shortlist_idx]
    short_gp = gp_primary[shortlist_idx]
    short_nn = nn_pred[shortlist_idx]
    short_cls = cls_mix[shortlist_idx]
    short_novelty = min_dist[shortlist_idx]
    short_boundary = boundary_weight[shortlist_idx]

    weights = HYBRID_WEIGHTS[strategy]
    stats = _hybrid_score_components(short_gp, short_nn, short_cls, short_novelty)
    hybrid_scores = np.array(
        [
            _hybrid_total_score(
                gp_value=short_gp[i],
                nn_value=short_nn[i],
                cls_value=short_cls[i],
                novelty_value=short_novelty[i],
                boundary_value=short_boundary[i],
                stats=stats,
                weights=weights,
            )
            for i in range(len(shortlist_idx))
        ],
        dtype=float,
    )

    ranked_short_idx = np.argsort(hybrid_scores)[::-1]
    start_points = short_candidates[ranked_short_idx[: min(15, len(ranked_short_idx))]]

    def total_score_single(point: np.ndarray) -> float:
        point_2d = np.asarray(point, dtype=float).reshape(1, -1)
        mu_s, sigma_s = gp.predict(point_2d, return_std=True)
        ei_s = expected_improvement(mu_s, sigma_s, best_y=best_y, xi=xi)
        ucb_s = upper_confidence_bound(mu_s, sigma_s, kappa=kappa)
        gp_s = float(ei_s[0] if acquisition == "ei" else ucb_s[0])
        nn_s = float(mlp.predict(point_2d)[0])
        p_good_s = float(0.5 * (logistic.predict_proba(point_2d)[0, 1] + svc.predict_proba(point_2d)[0, 1]))
        p_boundary_s = 1.0 - abs(p_good_s - 0.5) * 2.0
        cls_s = 0.75 * p_good_s + 0.25 * p_boundary_s
        novelty_s = float(np.min(np.linalg.norm(x - point_2d, axis=1)))
        bound_s = float(_boundary_distance(point_2d, low, high)[0])
        boundary_s = float(_boundary_weight(np.array([bound_s]), boundary_margin, floor=0.25)[0])
        return _hybrid_total_score(
            gp_value=gp_s,
            nn_value=nn_s,
            cls_value=cls_s,
            novelty_value=novelty_s,
            boundary_value=boundary_s,
            stats=stats,
            weights=weights,
        )

    refined = _refine_candidates(
        rng,
        x,
        start_points,
        total_score_single,
        low,
        high,
        strategy=strategy,
        novelty_floor=novelty_floor,
    )
    for point in start_points:
        refined.append((np.asarray(point, dtype=float), float(total_score_single(point))))

    chosen, chosen_score, boundary_override_used = _boundary_override(refined, low, high, boundary_margin)
    chosen_min_dist = float(np.min(np.linalg.norm(x - chosen.reshape(1, -1), axis=1)))
    chosen_bound_dist = float(_boundary_distance(chosen.reshape(1, -1), low, high)[0])
    chosen_mu, chosen_sigma = gp.predict(chosen.reshape(1, -1), return_std=True)
    chosen_ei = expected_improvement(chosen_mu, chosen_sigma, best_y=best_y, xi=xi)
    chosen_ucb = upper_confidence_bound(chosen_mu, chosen_sigma, kappa=kappa)
    chosen_p_good = float(
        0.5
        * (
            logistic.predict_proba(chosen.reshape(1, -1))[0, 1]
            + svc.predict_proba(chosen.reshape(1, -1))[0, 1]
        )
    )

    y_sorted_desc = np.argsort(y)[::-1]
    support_snapshots: List[Dict[str, Any]] = []
    for idx in support_boundary_sorted[:4]:
        support_snapshots.append(
            {
                "sample_index": int(idx),
                "output": float(y[idx]),
                "near_boundary_abs_decision": float(abs(train_decision[idx])),
            }
        )

    info: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_dims": int(dim),
        "strategy": strategy,
        "ucb_kappa": float(kappa),
        "best_y": best_y,
        "best_index": best_idx,
        "y_median": y_median,
        "y_std": float(np.std(y)),
        "z_best": float(z_best),
        "acquisition": acquisition,
        "classification_threshold": cls_threshold,
        "n_good_labels": int(np.sum(labels)),
        "n_bad_labels": int(len(labels) - np.sum(labels)),
        "support_vectors_count": int(len(support_indices)),
        "support_vectors_near_boundary": support_snapshots,
        "top_observed_outputs": [float(y[i]) for i in y_sorted_desc[:3]],
        "chosen_candidate_score": float(chosen_score),
        "chosen_candidate_min_dist": chosen_min_dist,
        "chosen_candidate_bound_dist": chosen_bound_dist,
        "chosen_candidate_ei": float(chosen_ei[0]),
        "chosen_candidate_ucb": float(chosen_ucb[0]),
        "chosen_candidate_p_good": chosen_p_good,
        "weights": dict(weights),
        "novelty_floor_applied": float(novelty_floor),
        "boundary_override_used": bool(boundary_override_used),
    }
    info.update(gp_info)
    info.update(regression_metrics)
    return chosen, info


def write_submission_outputs(
    out_dir: Path,
    raw_vectors: Dict[str, List[float]],
    portal_strings: Dict[str, str],
    debug_info: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    debug_label: str,
) -> None:
    _ensure_writable_dir(out_dir)

    lines = [_format_portal_line(i, np.array(raw_vectors[f"function_{i}"], dtype=float)) for i in range(1, 9)]
    (out_dir / f"{prefix}_portal_strings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / f"{prefix}_portal_strings.json").write_text(
        json.dumps({"portal_strings": portal_strings, "raw_vectors": raw_vectors}, indent=2),
        encoding="utf-8",
    )

    inputs_repr = "[" + ", ".join([repr(np.array(raw_vectors[f"function_{i}"])) for i in range(1, 9)]) + "]\n"
    (out_dir / f"{prefix}_inputs.txt").write_text(inputs_repr, encoding="utf-8")

    (out_dir / f"{prefix}_{debug_label}.json").write_text(
        json.dumps(debug_info, indent=2),
        encoding="utf-8",
    )


def build_gp_candidate_parser(
    *,
    description: str,
    seed_default: int,
    output_prefix: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to data root")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for portal strings")
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW, help="Lower bound (inclusive)")
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH, help="Upper bound (inclusive)")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.05,
        help="Soft margin to discourage extreme boundary inputs",
    )
    parser.add_argument(
        "--z-best-threshold",
        type=float,
        default=2.5,
        help="Use EI when z_best >= threshold, else UCB",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.96,
        help="UCB exploration parameter",
    )
    parser.add_argument(
        "--prefix",
        default=output_prefix,
        help=f"Output filename prefix (default {output_prefix}).",
    )
    return parser


def build_round_candidate_parser(
    *,
    description: str,
    inputs_default: Path,
    outputs_default: Path,
    seed_default: int,
    prefix_default: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--inputs-path",
        type=Path,
        default=inputs_default,
        help="Path to downloaded batched inputs file.",
    )
    parser.add_argument(
        "--outputs-path",
        type=Path,
        default=outputs_default,
        help="Path to downloaded batched outputs file.",
    )
    parser.add_argument(
        "--round-index",
        type=int,
        default=-1,
        help="Round batch index to ingest (default -1 = latest batch).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to initial_data directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for round artifacts.",
    )
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed.")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW, help="Lower input bound.")
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH, help="Upper input bound.")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.035,
        help="Soft margin used by candidate scoring near bounds.",
    )
    parser.add_argument(
        "--strategy",
        choices=["balanced", "explore", "exploit"],
        default="balanced",
        help="Candidate selection strategy profile.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="UCB kappa. If omitted, uses 1.25 (exploit), 1.96 (balanced), or 3.2 (explore).",
    )
    parser.add_argument(
        "--z-best-threshold",
        type=float,
        default=2.2,
        help="Use EI when z_best >= threshold, else UCB.",
    )
    parser.add_argument(
        "--prefix",
        default=prefix_default,
        help=f"Output filename prefix (default {prefix_default}).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip appending the round batch into initial_data.",
    )
    return parser


def run_gp_candidate_script(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_vectors: Dict[str, List[float]] = {}
    portal_strings: Dict[str, str] = {}
    debug_info: Dict[str, Dict[str, Any]] = {}

    for func_id in range(1, 9):
        func_dir = data_root / f"function_{func_id}"
        x = np.load(func_dir / "initial_inputs.npy")
        y = np.load(func_dir / "initial_outputs.npy").reshape(-1)
        candidate, info = choose_gp_candidate(
            x=x,
            y=y,
            rng=rng,
            low=args.low,
            high=args.high,
            boundary_margin=args.boundary_margin,
            z_best_threshold=args.z_best_threshold,
            kappa=args.kappa,
        )
        func_key = f"function_{func_id}"
        raw_vectors[func_key] = [float(v) for v in candidate.tolist()]
        portal_strings[func_key] = "-".join(f"{float(v):.6f}" for v in candidate)
        debug_info[func_key] = info

    write_submission_outputs(
        out_dir,
        raw_vectors,
        portal_strings,
        debug_info,
        prefix=args.prefix,
        debug_label="gp_debug",
    )


def run_round_candidate_script(
    args: argparse.Namespace,
    *,
    snapshot_filename: str,
) -> None:
    parsed = parse_latest_round(args.inputs_path, args.outputs_path, args.round_index)
    if args.skip_ingest:
        ingest_summary = {"skipped": True}
    else:
        ingest_summary = append_round_to_initial_data(
            data_root=args.data_root,
            round_inputs=parsed.round_inputs,
            round_outputs=parsed.round_outputs,
        )
        save_round_outputs_snapshot(parsed.round_outputs, args.out_dir, snapshot_filename)

    if args.kappa is not None:
        kappa = args.kappa
    elif args.strategy == "explore":
        kappa = 3.2
    elif args.strategy == "exploit":
        kappa = 1.25
    else:
        kappa = 1.96

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
            "kappa": kappa,
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
            kappa=kappa,
            z_best_threshold=args.z_best_threshold,
        )
        raw_vectors[func_key] = [float(v) for v in candidate.tolist()]
        portal_strings[func_key] = "-".join(f"{float(v):.6f}" for v in candidate)
        debug_info[func_key] = info

    write_submission_outputs(
        args.out_dir,
        raw_vectors,
        portal_strings,
        debug_info,
        prefix=args.prefix,
        debug_label="hybrid_debug",
    )
