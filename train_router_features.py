import json
import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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


def train_router(top_k: int = 10) -> None:
    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    selected_features = load_selected_features(top_k=top_k)
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    X, y = prepare_dataset(raw_data, selected_features)
    X, y = clean_data(X, y)

    print(f"Loaded {len(raw_data)} samples.")
    print(f"Failure rate: {y.mean():.2%} ({int(y.sum())} failures out of {len(y)})")

    if len(raw_data) < 5:
        print("Not enough data to split. Training on all data.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    print(f"Training Logistic Regression on {len(X_train)} samples...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    print("\n--- Threshold Optimization ---")
    probs = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    best_metrics = {}

    thresholds = np.arange(0.1, 1.0, 0.05)
    print(f"{'Threshold':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")

    for thresh in thresholds:
        y_pred_thresh = (probs >= thresh).astype(int)

        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        acc = accuracy_score(y_test, y_pred_thresh)

        if thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"{thresh:<10.2f} {f1:<10.4f} {p:<10.4f} {r:<10.4f} {acc:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {"precision": p, "recall": r, "accuracy": acc}

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"Best F1 score:  {best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")

    y_pred_best = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix (@ best threshold):")
    print("  [TN (True Negs)   FP (False Pos)] -> Actual Successs")
    print("  [FN (False Negs)  TP (True Pos)]  -> Actual Failures")
    print(cm)

    output_dict = {
        "selected_features": selected_features,
        "metrics": {
            "best_threshold": float(best_thresh),
            "f1_score": float(best_f1),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
        },
    }

    with open("router_features_results.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    print("\nDone! Results saved to router_features_results.json")


def main():
    train_router()


if __name__ == "__main__":
    main()
