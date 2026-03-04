from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import loguniform, norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "initial_data"
DEFAULT_OUT_DIR = REPO_ROOT / "deliverables" / "submissions"


@dataclass
class ParsedRoundData:
    round_inputs: List[np.ndarray]
    round_outputs: List[float]


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


def _parse_batch_file(path: Path) -> List[List[Any]]:
    text = path.read_text(encoding="utf-8")
    chunks = _extract_top_level_lists(text)
    parsed: List[List[Any]] = []
    eval_ctx = {"np": np, "array": np.array}
    for chunk in chunks:
        parsed.append(eval(chunk, eval_ctx, {}))  # noqa: S307 - trusted local artifact
    return parsed


def _parse_latest_round(inputs_path: Path, outputs_path: Path, round_index: int) -> ParsedRoundData:
    input_batches = _parse_batch_file(inputs_path)
    output_batches = _parse_batch_file(outputs_path)
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


def _save_round_03_outputs(round_outputs: List[float], out_dir: Path) -> None:
    values = ", ".join(f"np.float64({v!r})" for v in round_outputs)
    text = f"[{values}]\n"
    (out_dir / "round_03_outputs.txt").write_text(text, encoding="utf-8")


def _append_round_to_initial_data(
    data_root: Path,
    round_inputs: List[np.ndarray],
    round_outputs: List[float],
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


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = improvement * cdf + sigma * pdf
    ei[sigma <= 1e-9] = 0.0
    return ei


def _kernel_for_dim() -> object:
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-4, 10.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-9, 1.0))


def _fit_gp_with_search(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[GaussianProcessRegressor, Dict[str, float]]:
    dim = x.shape[1]
    base_kernel = _kernel_for_dim()
    gp = GaussianProcessRegressor(
        kernel=base_kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    param_distributions = {
        "kernel__k1__k1__constant_value": loguniform(1e-3, 1e3),
        "kernel__k1__k2__length_scale": loguniform(1e-4, 10.0),
        "kernel__k2__noise_level": loguniform(1e-9, 1.0),
    }
    n_iter = 8 if dim <= 4 else 6 if dim <= 6 else 4
    cv = 3 if len(y) >= 9 else 2
    search = RandomizedSearchCV(
        gp,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=int(rng.integers(0, 1_000_000)),
        n_jobs=1,
    )
    search.fit(x, y)
    info = {
        "best_score": float(search.best_score_),
        "best_constant": float(search.best_params_["kernel__k1__k1__constant_value"]),
        "best_length_scale": float(search.best_params_["kernel__k1__k2__length_scale"]),
        "best_noise_level": float(search.best_params_["kernel__k2__noise_level"]),
    }
    return search.best_estimator_, info


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
    model = TransformedTargetRegressor(
        regressor=mlp,
        transformer=StandardScaler(),
    )
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
        labels[np.argmax(y)] = 1

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
    dim = x.shape[1]
    n = len(y)
    n_splits = 3 if n >= 9 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    linear = Pipeline([("x_scale", StandardScaler()), ("model", LinearRegression())])
    svr = Pipeline([("x_scale", StandardScaler()), ("model", SVR(C=10.0, gamma="scale"))])
    mlp = _fit_mlp_regressor(x, y, seed=seed)

    scores_linear: List[float] = []
    scores_svr: List[float] = []
    scores_mlp: List[float] = []

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        linear.fit(x_train, y_train)
        pred_linear = linear.predict(x_test)
        scores_linear.append(float(r2_score(y_test, pred_linear)))

        svr.fit(x_train, y_train)
        pred_svr = svr.predict(x_test)
        scores_svr.append(float(r2_score(y_test, pred_svr)))

        mlp_fold = _fit_mlp_regressor(x_train, y_train, seed=seed + dim + len(train_idx))
        pred_mlp = mlp_fold.predict(x_test)
        scores_mlp.append(float(r2_score(y_test, pred_mlp)))

    return {
        "linear_r2_cv_mean": float(np.mean(scores_linear)),
        "svr_r2_cv_mean": float(np.mean(scores_svr)),
        "mlp_r2_cv_mean": float(np.mean(scores_mlp)),
    }


def _sample_candidates(
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
    else:
        n_global = 8000 if dim <= 4 else 10000 if dim <= 6 else 12000
        n_local_per_top = 450 if dim <= 4 else 350 if dim <= 6 else 250
        n_support = 200 if support_indices.size > 0 else 0
        sigma_local = 0.06 if dim <= 4 else 0.08 if dim <= 6 else 0.10

    global_samples = low + (high - low) * rng.random((n_global, dim))
    local_parts: List[np.ndarray] = [global_samples]

    for idx in top_idx:
        center = x[idx]
        local = center + rng.normal(0.0, sigma_local, size=(n_local_per_top, dim))
        local_parts.append(np.clip(local, low, high))

    for idx in support_indices[:3]:
        center = x[int(idx)]
        local = center + rng.normal(0.0, sigma_local * 0.75, size=(n_support, dim))
        local_parts.append(np.clip(local, low, high))

    local_parts.append(x[top_idx])
    return np.vstack(local_parts)


def _mlp_gradient(
    model: TransformedTargetRegressor,
    x_point: np.ndarray,
    low: float,
    high: float,
    step: float = 1e-3,
) -> np.ndarray:
    dim = x_point.shape[0]
    grad = np.zeros(dim, dtype=float)
    for j in range(dim):
        x_plus = x_point.copy()
        x_minus = x_point.copy()
        x_plus[j] = min(high, x_plus[j] + step)
        x_minus[j] = max(low, x_minus[j] - step)
        denom = x_plus[j] - x_minus[j]
        if denom <= 0:
            continue
        y_plus = float(model.predict(x_plus.reshape(1, -1))[0])
        y_minus = float(model.predict(x_minus.reshape(1, -1))[0])
        grad[j] = (y_plus - y_minus) / denom
    return grad


def _choose_candidate_for_function(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    boundary_margin: float,
    seed: int,
    strategy: str,
    kappa: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_samples, dim = x.shape
    y_std = float(np.std(y)) or 1.0
    y_median = float(np.median(y))
    best_idx = int(np.argmax(y))
    best_y = float(y[best_idx])
    z_best = (best_y - y_median) / y_std
    if strategy == "explore":
        acquisition = "ucb"
    else:
        acquisition = "ei" if z_best >= 2.2 else "ucb"

    gp, gp_info = _fit_gp_with_search(x, y, rng)
    mlp = _fit_mlp_regressor(x, y, seed=seed + dim)
    logistic, svc, labels, cls_threshold = _fit_classifiers(x, y, seed=seed + 100 + dim)
    regression_metrics = _evaluate_regression_models(x, y, seed=seed + 200 + dim)

    svc_model = svc.named_steps["svc"]
    train_decision = svc.decision_function(x)
    support_indices = np.asarray(svc_model.support_, dtype=int)
    boundary_order = np.argsort(np.abs(train_decision))
    support_boundary_sorted = [
        int(i) for i in boundary_order if int(i) in set(support_indices.tolist())
    ]
    if not support_boundary_sorted:
        support_boundary_sorted = [int(i) for i in support_indices.tolist()]
    support_for_sampling = np.asarray(support_boundary_sorted[:3], dtype=int)

    candidates = _sample_candidates(
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
    ei = _expected_improvement(mu, sigma, best_y=best_y, xi=xi)
    ucb = mu + kappa * sigma
    gp_primary = ei if acquisition == "ei" else ucb

    nn_pred = mlp.predict(candidates)
    p_good_log = logistic.predict_proba(candidates)[:, 1]
    p_good_svc = svc.predict_proba(candidates)[:, 1]
    p_good = 0.5 * (p_good_log + p_good_svc)
    p_boundary = 1.0 - np.abs(p_good - 0.5) * 2.0

    diff = candidates[:, None, :] - x[None, :, :]
    min_dist = np.min(np.linalg.norm(diff, axis=2), axis=1)

    dist_to_low = candidates - low
    dist_to_high = high - candidates
    min_bound_dist = np.minimum(dist_to_low, dist_to_high).min(axis=1)
    boundary_weight = 0.25 + 0.75 * np.clip(min_bound_dist / boundary_margin, 0.0, 1.0)

    gp_norm = (gp_primary - np.mean(gp_primary)) / (np.std(gp_primary) + 1e-9)
    nn_norm = (nn_pred - np.mean(nn_pred)) / (np.std(nn_pred) + 1e-9)
    cls_mix = 0.75 * p_good + 0.25 * p_boundary
    cls_norm = (cls_mix - np.mean(cls_mix)) / (np.std(cls_mix) + 1e-9)
    nov_norm = (min_dist - np.mean(min_dist)) / (np.std(min_dist) + 1e-9)

    if strategy == "explore":
        if z_best < 1.0:
            w_gp, w_nn, w_cls, w_nov = 0.30, 0.16, 0.15, 0.39
        elif z_best >= 2.8:
            w_gp, w_nn, w_cls, w_nov = 0.34, 0.20, 0.14, 0.32
        else:
            w_gp, w_nn, w_cls, w_nov = 0.32, 0.18, 0.15, 0.35
    else:
        if z_best < 1.0:
            w_gp, w_nn, w_cls, w_nov = 0.38, 0.24, 0.16, 0.22
        elif z_best >= 2.8:
            w_gp, w_nn, w_cls, w_nov = 0.52, 0.28, 0.15, 0.05
        else:
            w_gp, w_nn, w_cls, w_nov = 0.45, 0.28, 0.17, 0.10

    total = (w_gp * gp_norm) + (w_nn * nn_norm) + (w_cls * cls_norm) + (w_nov * nov_norm)
    total = total * boundary_weight

    valid_mask = min_dist > 1e-5
    novelty_floor = 0.0
    if strategy == "explore":
        # Enforce a higher novelty floor in exploration mode.
        novelty_floor = float(np.quantile(min_dist, 0.65))
        valid_mask = valid_mask & (min_dist >= novelty_floor)
        if not np.any(valid_mask):
            novelty_floor = float(np.quantile(min_dist, 0.50))
            valid_mask = min_dist >= novelty_floor
    scored = np.where(valid_mask, total, -np.inf)
    top_idx = np.argsort(scored)[-15:]
    top_idx = top_idx[np.isfinite(scored[top_idx])]
    if top_idx.size == 0:
        top_idx = np.array([int(np.argmax(total))], dtype=int)

    def total_score_single(point: np.ndarray) -> float:
        point_2d = point.reshape(1, -1)
        mu_s, sigma_s = gp.predict(point_2d, return_std=True)
        ei_s = _expected_improvement(mu_s, sigma_s, best_y=best_y, xi=xi)
        ucb_s = mu_s + kappa * sigma_s
        gp_s = float(ei_s[0] if acquisition == "ei" else ucb_s[0])
        nn_s = float(mlp.predict(point_2d)[0])
        p_good_s = float(0.5 * (logistic.predict_proba(point_2d)[0, 1] + svc.predict_proba(point_2d)[0, 1]))
        p_boundary_s = 1.0 - abs(p_good_s - 0.5) * 2.0
        cls_s = 0.75 * p_good_s + 0.25 * p_boundary_s
        min_dist_s = float(np.min(np.linalg.norm(x - point_2d, axis=1)))
        min_bound_dist_s = float(np.minimum(point - low, high - point).min())
        bw = 0.25 + 0.75 * np.clip(min_bound_dist_s / boundary_margin, 0.0, 1.0)

        gp_s = (gp_s - float(np.mean(gp_primary))) / (float(np.std(gp_primary)) + 1e-9)
        nn_s = (nn_s - float(np.mean(nn_pred))) / (float(np.std(nn_pred)) + 1e-9)
        cls_s = (cls_s - float(np.mean(cls_mix))) / (float(np.std(cls_mix)) + 1e-9)
        nov_s = (min_dist_s - float(np.mean(min_dist))) / (float(np.std(min_dist)) + 1e-9)
        return float((w_gp * gp_s + w_nn * nn_s + w_cls * cls_s + w_nov * nov_s) * bw)

    best_candidate = candidates[int(top_idx[-1])].copy()
    best_score = float(total_score_single(best_candidate))
    best_grad = _mlp_gradient(mlp, best_candidate, low=low, high=high)

    for idx in top_idx[::-1]:
        current = candidates[int(idx)].copy()
        current_score = float(total_score_single(current))
        step = 0.03 if dim <= 4 else 0.025 if dim <= 6 else 0.02
        if strategy == "explore":
            step *= 0.8
        last_grad = np.zeros(dim, dtype=float)

        for _ in range(8):
            grad = _mlp_gradient(mlp, current, low=low, high=high)
            last_grad = grad
            norm_grad = float(np.linalg.norm(grad))
            if norm_grad < 1e-10:
                break
            trial = np.clip(current + step * (grad / norm_grad), low, high)
            if float(np.min(np.linalg.norm(x - trial.reshape(1, -1), axis=1))) <= 1e-5:
                step *= 0.5
                if step < 0.004:
                    break
                continue
            trial_min_dist = float(np.min(np.linalg.norm(x - trial.reshape(1, -1), axis=1)))
            if strategy == "explore" and trial_min_dist < novelty_floor:
                step *= 0.5
                if step < 0.004:
                    break
                continue
            trial_score = float(total_score_single(trial))
            if trial_score > current_score:
                current = trial
                current_score = trial_score
            else:
                step *= 0.5
                if step < 0.004:
                    break

        if current_score > best_score:
            best_candidate = current
            best_score = current_score
            best_grad = last_grad

    chosen = np.clip(best_candidate, low, high)
    chosen_min_dist = float(np.min(np.linalg.norm(x - chosen.reshape(1, -1), axis=1)))
    chosen_mu, chosen_sigma = gp.predict(chosen.reshape(1, -1), return_std=True)
    chosen_ei = _expected_improvement(chosen_mu, chosen_sigma, best_y=best_y, xi=xi)
    chosen_ucb = chosen_mu + 1.96 * chosen_sigma
    chosen_p_good = float(
        0.5
        * (
            logistic.predict_proba(chosen.reshape(1, -1))[0, 1]
            + svc.predict_proba(chosen.reshape(1, -1))[0, 1]
        )
    )

    abs_grad = np.abs(best_grad)
    grad_rank_idx = np.argsort(abs_grad)[::-1]
    top_grad_dims = [int(i) for i in grad_rank_idx[: min(3, dim)]]

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
        "ucb_kappa": kappa,
        "best_y": best_y,
        "best_index": best_idx,
        "y_median": y_median,
        "y_std": y_std,
        "z_best": float(z_best),
        "acquisition": acquisition,
        "classification_threshold": cls_threshold,
        "n_good_labels": int(np.sum(labels)),
        "n_bad_labels": int(len(labels) - np.sum(labels)),
        "support_vectors_count": int(len(support_indices)),
        "support_vectors_near_boundary": support_snapshots,
        "top_observed_outputs": [float(y[i]) for i in y_sorted_desc[:3]],
        "chosen_candidate_score": best_score,
        "chosen_candidate_min_dist": chosen_min_dist,
        "chosen_candidate_ei": float(chosen_ei[0]),
        "chosen_candidate_ucb": float(chosen_ucb[0]),
        "chosen_candidate_p_good": chosen_p_good,
        "chosen_candidate_grad_norm": float(np.linalg.norm(best_grad)),
        "chosen_candidate_top_grad_dims_1_based": [d + 1 for d in top_grad_dims],
        "chosen_candidate_top_grad_values": [float(best_grad[d]) for d in top_grad_dims],
        "weights": {
            "gp": w_gp,
            "nn": w_nn,
            "classification": w_cls,
            "novelty": w_nov,
        },
        "novelty_floor_applied": novelty_floor,
    }
    info.update(gp_info)
    info.update(regression_metrics)
    return chosen, info


def _format_portal_line(function_id: int, vector: np.ndarray) -> str:
    return f"function_{function_id}: " + "-".join(f"{float(v):.6f}" for v in vector)


def _write_round_04_outputs(
    out_dir: Path,
    raw_vectors: Dict[str, List[float]],
    portal_strings: Dict[str, str],
    debug_info: Dict[str, Dict[str, Any]],
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [_format_portal_line(i, np.array(raw_vectors[f"function_{i}"], dtype=float)) for i in range(1, 9)]
    (out_dir / f"{prefix}_portal_strings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / f"{prefix}_portal_strings.json").write_text(
        json.dumps({"portal_strings": portal_strings, "raw_vectors": raw_vectors}, indent=2),
        encoding="utf-8",
    )

    inputs_repr = "[" + ", ".join([repr(np.array(raw_vectors[f"function_{i}"])) for i in range(1, 9)]) + "]\n"
    (out_dir / f"{prefix}_inputs.txt").write_text(inputs_repr, encoding="utf-8")

    (out_dir / f"{prefix}_hybrid_debug.json").write_text(
        json.dumps(debug_info, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest latest round outputs and propose Round 04 candidates (GP + NN hybrid)."
    )
    parser.add_argument(
        "--inputs-path",
        type=Path,
        default=Path(r"c:\Users\tom_m\Downloads\inputs.txt"),
        help="Path to downloaded batched inputs file.",
    )
    parser.add_argument(
        "--outputs-path",
        type=Path,
        default=Path(r"c:\Users\tom_m\Downloads\outputs.txt"),
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
    parser.add_argument("--seed", type=int, default=20260218, help="Random seed.")
    parser.add_argument("--low", type=float, default=0.001, help="Lower input bound.")
    parser.add_argument("--high", type=float, default=0.98, help="Upper input bound.")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.035,
        help="Soft margin used by candidate scoring near bounds.",
    )
    parser.add_argument(
        "--strategy",
        choices=["balanced", "explore"],
        default="balanced",
        help="Candidate selection strategy profile.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="UCB kappa. If omitted, uses 1.96 (balanced) or 3.2 (explore).",
    )
    parser.add_argument(
        "--prefix",
        default="round_04",
        help="Output filename prefix (e.g., round_04_explore).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip appending the round batch into initial_data.",
    )
    args = parser.parse_args()

    parsed = _parse_latest_round(args.inputs_path, args.outputs_path, args.round_index)
    if args.skip_ingest:
        ingest_summary = {"skipped": True}
    else:
        ingest_summary = _append_round_to_initial_data(
            data_root=args.data_root,
            round_inputs=parsed.round_inputs,
            round_outputs=parsed.round_outputs,
        )
        _save_round_03_outputs(parsed.round_outputs, args.out_dir)

    kappa = args.kappa if args.kappa is not None else (3.2 if args.strategy == "explore" else 1.96)

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
            "prefix": args.prefix,
            "skip_ingest": bool(args.skip_ingest),
        },
    }

    for func_id in range(1, 9):
        func_key = f"function_{func_id}"
        x = np.load(args.data_root / func_key / "initial_inputs.npy")
        y = np.load(args.data_root / func_key / "initial_outputs.npy").reshape(-1)
        candidate, info = _choose_candidate_for_function(
            x=x,
            y=y,
            rng=rng,
            low=args.low,
            high=args.high,
            boundary_margin=args.boundary_margin,
            seed=args.seed + func_id * 13,
            strategy=args.strategy,
            kappa=kappa,
        )
        raw_vectors[func_key] = [float(v) for v in candidate.tolist()]
        portal_strings[func_key] = "-".join(f"{float(v):.6f}" for v in candidate)
        debug_info[func_key] = info

    _write_round_04_outputs(args.out_dir, raw_vectors, portal_strings, debug_info, prefix=args.prefix)


if __name__ == "__main__":
    main()
