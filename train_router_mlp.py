import json
import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_answer(text):
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
    if "is_correct" in entry:
        return entry["is_correct"]
    pred_val = extract_answer(entry.get("generated_text", ""))
    truth_val = extract_answer(entry.get("ground_truth", ""))
    if pred_val is None or truth_val is None:
        return False
    return abs(pred_val - truth_val) < 1e-4


def load_data(filename="trajectory_data_f32.jsonl"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_features(data):
    X, y = [], []
    for entry in data:
        features = entry["cosine_sims"] + [entry["final_entropy"]]
        X.append(features)
        is_correct = compute_correctness(entry)
        y.append(1 if not is_correct else 0)
    return np.array(X), np.array(y)


def clean_data(X, y):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y


def balance_train_set(X_train, y_train, random_state=42):
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


def find_best_threshold(probs, y_test):
    best_thresh, best_f1 = 0.5, -1.0
    best_metrics = {}
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_test, y_pred),
            }
    return best_thresh, best_f1, best_metrics


def train_router_mlp():
    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    X, y = prepare_features(raw_data)
    X, y = clean_data(X, y)

    print(f"Loaded {len(raw_data)} samples.")
    print(f"Original failure rate: {y.mean():.2%} ({int(y.sum())} failures out of {len(y)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_bal, y_train_bal = balance_train_set(X_train, y_train)

    # MLP needs feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train (original): {len(X_train)} | Test: {len(X_test)}")
    print(
        f"Train (balanced): {len(X_train_bal)} with "
        f"{int((y_train_bal == 1).sum())} failures and {int((y_train_bal == 0).sum())} successes"
    )

    configs = [
        ("Shallow (64)",        (64,)),
        ("Medium (128-64)",     (128, 64)),
        ("Deep (256-128-64)",   (256, 128, 64)),
    ]

    print("\n--- MLP Architecture Comparison ---")
    print(f"{'Architecture':<26} {'F1':<8} {'Precision':<11} {'Recall':<9} {'Accuracy':<10} {'ROC AUC':<9} {'PR AUC'}")

    best_model_info = None
    best_overall_f1 = -1.0

    for name, hidden in configs:
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            learning_rate_init=1e-3,
        )
        model.fit(X_train_scaled, y_train_bal)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        thresh, f1, metrics = find_best_threshold(probs, y_test)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        print(
            f"{name:<26} {f1:<8.4f} {metrics['precision']:<11.4f} "
            f"{metrics['recall']:<9.4f} {metrics['accuracy']:<10.4f} "
            f"{roc_auc:<9.4f} {pr_auc:.4f}  (thresh={thresh:.2f})"
        )

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_model_info = {
                "name": name, "hidden": hidden, "model": model,
                "probs": probs, "thresh": thresh, "f1": f1,
                "metrics": metrics, "roc_auc": roc_auc, "pr_auc": pr_auc,
            }

    info = best_model_info
    y_pred_best = (info["probs"] >= info["thresh"]).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\n--- Best Architecture: {info['name']} ---")
    print(f"Best threshold: {info['thresh']:.2f}")
    print(f"F1 Score:       {info['f1']:.4f}")
    print(f"Precision:      {info['metrics']['precision']:.4f}")
    print(f"Recall:         {info['metrics']['recall']:.4f}")
    print(f"Accuracy:       {info['metrics']['accuracy']:.4f}")
    print(f"ROC AUC:        {info['roc_auc']:.4f}")
    print(f"PR AUC:         {info['pr_auc']:.4f}")
    print("\nConfusion Matrix (@ best threshold):")
    print("  [TN (True Negs)   FP (False Pos)] -> Actual Successes")
    print("  [FN (False Negs)  TP (True Pos)]  -> Actual Failures")
    print(cm)

    output_dict = {
        "model": "MLP",
        "best_architecture": info["name"],
        "hidden_layer_sizes": list(info["hidden"]),
        "split": {
            "train_original": int(len(X_train)),
            "train_balanced": int(len(X_train_bal)),
            "test": int(len(X_test)),
        },
        "metrics": {
            "best_threshold": float(info["thresh"]),
            "f1_score": float(info["f1"]),
            "precision": float(info["metrics"]["precision"]),
            "recall": float(info["metrics"]["recall"]),
            "accuracy": float(info["metrics"]["accuracy"]),
            "roc_auc": float(info["roc_auc"]),
            "pr_auc": float(info["pr_auc"]),
        },
        "confusion_matrix": cm.tolist(),
    }

    with open("router_mlp_results.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)

    print("\nDone! Results saved to router_mlp_results.json")


if __name__ == "__main__":
    train_router_mlp()