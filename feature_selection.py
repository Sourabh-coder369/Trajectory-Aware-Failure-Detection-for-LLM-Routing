import csv
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureResult:
    name: str
    l1_importance: float
    perm_importance: float
    stability: float


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


def load_data(filename: str = "data/trajectory_data_f32.jsonl") -> List[Dict]:
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


def prepare_dataset(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    feature_names: List[str] = []

    for entry in data:
        feats = extract_features(entry.get("cosine_sims", []), entry.get("final_entropy", 0.0))
        if not feature_names:
            feature_names = list(feats.keys())
        rows.append([feats[name] for name in feature_names])

        is_correct = compute_correctness(entry)
        labels.append(0 if is_correct else 1)

    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64), feature_names


def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def filter_low_variance(X: np.ndarray, feature_names: List[str], threshold: float = 1e-6):
    variances = np.var(X, axis=0)
    keep_mask = variances >= threshold
    kept_names = [name for name, keep in zip(feature_names, keep_mask) if keep]
    return X[:, keep_mask], kept_names


def filter_high_correlation(X: np.ndarray, feature_names: List[str], threshold: float = 0.90):
    if X.shape[1] <= 1:
        return X, feature_names

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    to_drop = set()

    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            if abs(corr[i, j]) > threshold and j not in to_drop:
                to_drop.add(j)

    keep_indices = [i for i in range(len(feature_names)) if i not in to_drop]
    kept_names = [feature_names[i] for i in keep_indices]
    return X[:, keep_indices], kept_names


def select_features(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[FeatureResult]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    l1_scores = np.zeros(len(feature_names), dtype=np.float64)
    perm_scores = np.zeros(len(feature_names), dtype=np.float64)
    stability_counts = np.zeros(len(feature_names), dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                C=0.5,
            )),
        ])
        model.fit(X_train, y_train)

        coef = model.named_steps["clf"].coef_[0]
        l1_scores += np.abs(coef)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="f1",
        )
        perm_scores += perm.importances_mean

        top_k = max(1, len(feature_names) // 4)
        top_idx = np.argsort(perm.importances_mean)[-top_k:]
        stability_counts[top_idx] += 1

    l1_scores /= skf.get_n_splits()
    perm_scores /= skf.get_n_splits()
    stability = stability_counts / skf.get_n_splits()

    results = []
    for name, l1, perm_imp, stab in zip(feature_names, l1_scores, perm_scores, stability):
        results.append(FeatureResult(name=name, l1_importance=float(l1), perm_importance=float(perm_imp), stability=float(stab)))

    results.sort(key=lambda r: (r.stability, r.perm_importance, r.l1_importance), reverse=True)
    return results


def save_results(results: List[FeatureResult], output_json: str, output_csv: str) -> None:
    json_data = [
        {
            "feature": r.name,
            "l1_importance": r.l1_importance,
            "perm_importance": r.perm_importance,
            "stability": r.stability,
        }
        for r in results
    ]
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "l1_importance", "perm_importance", "stability"])
        writer.writeheader()
        for row in json_data:
            writer.writerow(row)


def main():
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data found.")
        return

    X, y, feature_names = prepare_dataset(data)
    X, y = clean_data(X, y)

    print(f"Samples: {len(y)} | Failures: {int(y.sum())} ({y.mean():.2%})")
    X, feature_names = filter_low_variance(X, feature_names)
    X, feature_names = filter_high_correlation(X, feature_names)

    print(f"Features after filtering: {len(feature_names)}")
    results = select_features(X, y, feature_names)
    save_results(results, "feature_selection_results.json", "feature_selection_results.csv")

    print("Top features:")
    for r in results[:10]:
        print(f"  {r.name}: stability={r.stability:.2f}, perm={r.perm_importance:.4f}, l1={r.l1_importance:.4f}")
    print("\nSaved: feature_selection_results.json, feature_selection_results.csv")


if __name__ == "__main__":
    main()
