import json
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def extract_answer(text):
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


def compute_correctness(entry):
    """Determine if the model's answer is correct by comparing
    the extracted numeric answer from generated_text vs ground_truth."""
    if "is_correct" in entry:
        return entry["is_correct"]

    pred_val = extract_answer(entry.get("generated_text", ""))
    truth_val = extract_answer(entry.get("ground_truth", ""))

    if pred_val is None or truth_val is None:
        return False

    return abs(pred_val - truth_val) < 1e-4


def load_data(filename="data/trajectory_data_f32.jsonl"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_features(data):
    X = []
    y = []
    for entry in data:
        features = entry["cosine_sims"] + [entry["final_entropy"]]
        X.append(features)

        # Label=1 means failure, Label=0 means success
        is_correct = compute_correctness(entry)
        y.append(1 if not is_correct else 0)

    return np.array(X), np.array(y)


def clean_data(X, y):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def balance_train_set(X_train, y_train, random_state=42):
    """Create a balanced training set via random undersampling."""
    idx_fail = np.where(y_train == 1)[0]
    idx_success = np.where(y_train == 0)[0]

    if len(idx_fail) == 0 or len(idx_success) == 0:
        return X_train, y_train

    n = min(len(idx_fail), len(idx_success))
    rng = np.random.default_rng(random_state)

    fail_sel = rng.choice(idx_fail, size=n, replace=False)
    success_sel = rng.choice(idx_success, size=n, replace=False)

    keep_idx = np.concatenate([fail_sel, success_sel])
    rng.shuffle(keep_idx)

    return X_train[keep_idx], y_train[keep_idx]


def train_router_balanced():
    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    X, y = prepare_features(raw_data)
    X, y = clean_data(X, y)

    print(f"Loaded {len(raw_data)} samples.")
    print(f"Original failure rate: {y.mean():.2%} ({int(y.sum())} failures out of {len(y)})")

    if len(raw_data) < 5:
        print("Not enough data to split. Training on all data.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    X_train_bal, y_train_bal = balance_train_set(X_train, y_train, random_state=42)

    print(f"Train (original): {len(X_train)} | Test: {len(X_test)}")
    print(
        f"Train (balanced): {len(X_train_bal)} with "
        f"{int((y_train_bal == 1).sum())} failures and {int((y_train_bal == 0).sum())} successes"
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Threshold Optimization ---")
    best_thresh = 0.5
    best_f1 = -1.0
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

    y_pred_best = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"Best F1 score:  {best_f1:.4f}")
    print(f"Precision:      {best_metrics['precision']:.4f}")
    print(f"Recall:         {best_metrics['recall']:.4f}")
    print(f"Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"ROC AUC:        {roc_auc:.4f}")
    print(f"PR AUC:         {pr_auc:.4f}")

    print("\nConfusion Matrix (@ best threshold):")
    print("  [TN (True Negs)   FP (False Pos)] -> Actual Successes")
    print("  [FN (False Negs)  TP (True Pos)]  -> Actual Failures")
    print(cm)

    coefs = model.coef_[0]
    output_dict = {
        "split": {
            "train_original": int(len(X_train)),
            "train_balanced": int(len(X_train_bal)),
            "test": int(len(X_test)),
            "train_balanced_failures": int((y_train_bal == 1).sum()),
            "train_balanced_successes": int((y_train_bal == 0).sum()),
        },
        "metrics": {
            "best_threshold": float(best_thresh),
            "f1_score": float(best_f1),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "accuracy": float(best_metrics["accuracy"]),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "coefficients": {
            "intercept": float(model.intercept_[0]),
            "final_entropy_coef": float(coefs[-1]),
            "cosine_sims_coefs": [float(c) for c in coefs[:-1]],
        },
        "confusion_matrix": cm.tolist(),
    }

    with open("results/router_balanced_results.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)

    print("\nDone! Results saved to router_balanced_results.json")


if __name__ == "__main__":
    train_router_balanced()
