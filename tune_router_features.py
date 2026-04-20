import json
import re
from typing import Dict, List, Tuple

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_answer(text: str):
    """Extract the final numeric answer from text.
    For ground_truth: looks for #### pattern first (GSM8K format).
    For generated_text: extracts the last number found.
    """
    if text is None:
        return None

    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))

    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))


def compute_correctness(entry: Dict) -> bool:
    """Determine if the model's answer is correct."""
    if "is_correct" in entry:
        return entry["is_correct"]

    pred_val = extract_answer(entry.get("generated_text", ""))
    truth_val = extract_answer(entry.get("ground_truth", ""))

    if pred_val is None or truth_val is None:
        return False

    return abs(pred_val - truth_val) < 1e-4


def load_data(filename: str = "trajectory_data_f32.jsonl") -> List[Dict]:
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _safe_autocorr(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return 0.0
    v1 = values[:-lag]
    v2 = values[lag:]
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])


def _segment_means(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    third = max(values.size // 3, 1)
    early = values[:third]
    mid = values[third: 2 * third] if values.size >= 2 * third else values[third:]
    late = values[2 * third:] if values.size > 2 * third else values[-third:]
    return _safe_mean(early), _safe_mean(mid), _safe_mean(late)


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    y = values.astype(np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _second_diff_energy(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    second_diff = np.diff(values, n=2)
    return float(np.mean(second_diff ** 2))


def _sign_change_count(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    return float(np.sum(signs[1:] * signs[:-1] < 0))


def extract_features(cosine_sims: List[float], final_entropy: float) -> Dict[str, float]:
    x = np.array(cosine_sims, dtype=np.float64)
    features: Dict[str, float] = {}

    features["cos_mean"] = _safe_mean(x)
    features["cos_std"] = _safe_std(x)
    features["cos_min"] = float(np.min(x)) if x.size else 0.0
    features["cos_max"] = float(np.max(x)) if x.size else 0.0
    q10 = _safe_quantile(x, 0.10)
    q50 = _safe_quantile(x, 0.50)
    q90 = _safe_quantile(x, 0.90)
    features["cos_q10"] = q10
    features["cos_q50"] = q50
    features["cos_q90"] = q90
    features["cos_iqr"] = q90 - q10

    if x.size >= 2:
        diffs = np.diff(x)
        features["cos_total_variation"] = float(np.sum(np.abs(diffs)))
        features["cos_mean_abs_diff"] = float(np.mean(np.abs(diffs)))
        features["cos_max_drop"] = float(np.min(diffs))
        features["cos_max_drop_idx"] = float(np.argmin(diffs))
    else:
        features["cos_total_variation"] = 0.0
        features["cos_mean_abs_diff"] = 0.0
        features["cos_max_drop"] = 0.0
        features["cos_max_drop_idx"] = 0.0

    features["cos_second_diff_energy"] = _second_diff_energy(x)
    features["cos_slope"] = _linear_slope(x)

    early_mean, mid_mean, late_mean = _segment_means(x)
    features["cos_early_mean"] = early_mean
    features["cos_mid_mean"] = mid_mean
    features["cos_late_mean"] = late_mean
    features["cos_late_minus_early"] = late_mean - early_mean

    features["cos_sign_change_count"] = _sign_change_count(x)
    features["cos_autocorr_lag1"] = _safe_autocorr(x, 1)
    features["cos_autocorr_lag2"] = _safe_autocorr(x, 2)
    features["cos_autocorr_lag3"] = _safe_autocorr(x, 3)

    features["cos_area"] = _safe_mean(x)
    features["final_entropy"] = float(final_entropy)

    features["entropy_x_roughness"] = float(final_entropy) * features["cos_mean_abs_diff"]
    features["entropy_x_late_mean"] = float(final_entropy) * features["cos_late_mean"]
    features["entropy_x_max_drop"] = float(final_entropy) * features["cos_max_drop"]

    return features


def load_selected_features(
    results_path: str = "feature_selection_results.json",
    top_k: int = 10,
) -> List[str]:
    with open(results_path, "r") as f:
        data = json.load(f)
    return [row["feature"] for row in data[:top_k]]


def prepare_dataset(data: List[Dict], selected_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    labels: List[int] = []

    for entry in data:
        feats = extract_features(entry.get("cosine_sims", []), entry.get("final_entropy", 0.0))
        rows.append([feats.get(name, 0.0) for name in selected_features])

        is_correct = compute_correctness(entry)
        labels.append(0 if is_correct else 1)

    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def cv_f1_with_threshold_tuning(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
    cv: StratifiedKFold,
) -> Dict[str, float]:
    best_f1_scores = []
    best_thresholds = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        probs = estimator.predict_proba(X_test)[:, 1]

        fold_best_f1 = -1.0
        fold_best_thresh = 0.5

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            score = f1_score(y_test, preds, zero_division=0)
            if score > fold_best_f1:
                fold_best_f1 = score
                fold_best_thresh = thresh

        best_f1_scores.append(fold_best_f1)
        best_thresholds.append(fold_best_thresh)

    return {
        "cv_f1_mean": float(np.mean(best_f1_scores)),
        "cv_f1_std": float(np.std(best_f1_scores)),
        "cv_best_threshold_mean": float(np.mean(best_thresholds)),
        "cv_best_threshold_std": float(np.std(best_thresholds)),
    }


def tune_logistic_regression(X: np.ndarray, y: np.ndarray) -> Dict:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        )),
    ])

    param_grid = {
        "clf__C": np.logspace(-3, 2, 20),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=100,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)

    cv_threshold_results = cv_f1_with_threshold_tuning(
        estimator=search.best_estimator_,
        X=X,
        y=y,
        thresholds=np.arange(0.1, 1.0, 0.05),
        cv=cv,
    )

    return {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_threshold_f1": cv_threshold_results,
    }


def tune_xgboost(X: np.ndarray, y: np.ndarray) -> Dict:
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_weight,
        random_state=42,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)

    cv_threshold_results = cv_f1_with_threshold_tuning(
        estimator=search.best_estimator_,
        X=X,
        y=y,
        thresholds=np.arange(0.1, 1.0, 0.05),
        cv=cv,
    )

    return {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_threshold_f1": cv_threshold_results,
    }


def main():
    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    selected_features = load_selected_features(top_k=10)
    X, y = prepare_dataset(raw_data, selected_features)
    X, y = clean_data(X, y)

    print(f"Samples: {len(y)} | Failures: {int(y.sum())} ({y.mean():.2%})")
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    print("\nTuning Logistic Regression...")
    lr_results = tune_logistic_regression(X, y)

    print("\nTuning XGBoost...")
    xgb_results = tune_xgboost(X, y)

    results = {
        "selected_features": selected_features,
        "logistic_regression": lr_results,
        "xgboost": xgb_results,
    }

    with open("router_features_tuning_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nDone! Saved to router_features_tuning_results.json")


if __name__ == "__main__":
    main()
