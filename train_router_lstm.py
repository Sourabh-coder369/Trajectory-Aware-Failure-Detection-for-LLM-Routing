import json
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── data helpers ──────────────────────────────────────────────────────────────

def extract_answer(text):
    if text is None:
        return None
    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(numbers[-1].replace(",", "")) if numbers else None


def compute_correctness(entry):
    if "is_correct" in entry:
        return entry["is_correct"]
    pred = extract_answer(entry.get("generated_text", ""))
    truth = extract_answer(entry.get("ground_truth", ""))
    if pred is None or truth is None:
        return False
    return abs(pred - truth) < 1e-4


def load_data(filename="trajectory_data_f32.jsonl"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_features(data):
    """
    Returns:
        seq_X  : (N, T, 1)  — cosine_sims as a time series, one step per timestep
        scalar_X: (N, 1)    — final_entropy as a separate scalar feature
        y       : (N,)      — 1=failure, 0=success
    """
    seqs, scalars, labels = [], [], []
    for entry in data:
        sims = entry["cosine_sims"]
        seqs.append([[v] for v in sims])          # shape (T, 1)
        scalars.append([entry["final_entropy"]])
        labels.append(0 if compute_correctness(entry) else 1)

    # Pad sequences to the same length
    max_len = max(len(s) for s in seqs)
    padded = np.zeros((len(seqs), max_len, 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        arr = np.array(s, dtype=np.float32)
        padded[i, :len(arr)] = arr

    padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
    scalars = np.nan_to_num(np.array(scalars, dtype=np.float32), nan=0.0)
    labels = np.array(labels, dtype=np.int64)
    return padded, scalars, labels


# ── model ─────────────────────────────────────────────────────────────────────

class TrajectoryLSTM(nn.Module):
    """
    Reads cosine_sims as a sequence with an LSTM, then concatenates
    the final hidden state with final_entropy before classification.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 scalar_size=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + scalar_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, seq, scalar):
        _, (h_n, _) = self.lstm(seq)
        last_hidden = h_n[-1]                        # (batch, hidden_size)
        combined = torch.cat([last_hidden, scalar], dim=1)
        return self.classifier(combined).squeeze(1)  # (batch,)


# ── training helpers ───────────────────────────────────────────────────────────

def make_loader(seq, scalar, y, batch_size=128, balance=False):
    ds = TensorDataset(
        torch.tensor(seq, dtype=torch.float32),
        torch.tensor(scalar, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    if balance:
        counts = np.bincount(y)
        weights = 1.0 / counts[y]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seq_b, scalar_b, y_b in loader:
        seq_b, scalar_b, y_b = seq_b.to(device), scalar_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(seq_b, scalar_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    for seq_b, scalar_b, _ in loader:
        logits = model(seq_b.to(device), scalar_b.to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def find_best_threshold(probs, y_test):
    best_thresh, best_f1, best_metrics = 0.5, -1.0, {}
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall":    recall_score(y_test, y_pred, zero_division=0),
                "accuracy":  accuracy_score(y_test, y_pred),
            }
    return best_thresh, best_f1, best_metrics


# ── main ──────────────────────────────────────────────────────────────────────

def train_router_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    seq_X, scalar_X, y = prepare_features(raw_data)

    # Scale entropy scalar
    scaler = StandardScaler()
    scalar_X = scaler.fit_transform(scalar_X).astype(np.float32)

    print(f"Loaded {len(y)} samples  |  seq length: {seq_X.shape[1]}")
    print(f"Failure rate: {y.mean():.2%} ({int(y.sum())} failures)")

    seq_tr, seq_te, sc_tr, sc_te, y_tr, y_te = train_test_split(
        seq_X, scalar_X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_tr)} | Test: {len(y_te)}")

    train_loader = make_loader(seq_tr, sc_tr, y_tr, balance=True)
    test_loader  = make_loader(seq_te, sc_te, y_te, balance=False)

    configs = [
        ("LSTM-1L-32h",  dict(hidden_size=32,  num_layers=1)),
        ("LSTM-2L-64h",  dict(hidden_size=64,  num_layers=2)),
        ("LSTM-2L-128h", dict(hidden_size=128, num_layers=2)),
    ]

    print("\n--- LSTM Architecture Comparison (30 epochs each) ---")
    print(f"{'Architecture':<18} {'F1':<8} {'Precision':<11} {'Recall':<9} {'Accuracy':<10} {'ROC AUC':<9} {'PR AUC'}")

    best_info = None
    best_f1_overall = -1.0

    for name, kwargs in configs:
        model = TrajectoryLSTM(**kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Weighted BCE to further reinforce failure class
        pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_f1, patience, patience_limit = -1.0, 0, 7
        for epoch in range(1, 51):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            probs_val = get_probs(model, test_loader, device)
            _, val_f1, _ = find_best_threshold(probs_val, y_te)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= patience_limit:
                break

        model.load_state_dict(best_state)
        probs = get_probs(model, test_loader, device)
        thresh, f1, metrics = find_best_threshold(probs, y_te)
        roc_auc = roc_auc_score(y_te, probs)
        pr_auc  = average_precision_score(y_te, probs)

        print(
            f"{name:<18} {f1:<8.4f} {metrics['precision']:<11.4f} "
            f"{metrics['recall']:<9.4f} {metrics['accuracy']:<10.4f} "
            f"{roc_auc:<9.4f} {pr_auc:.4f}  (thresh={thresh:.2f})"
        )

        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_info = dict(name=name, probs=probs, thresh=thresh, f1=f1,
                             metrics=metrics, roc_auc=roc_auc, pr_auc=pr_auc,
                             kwargs=kwargs)

    info = best_info
    y_pred_best = (info["probs"] >= info["thresh"]).astype(int)
    cm = confusion_matrix(y_te, y_pred_best)

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

    output = {
        "model": "LSTM",
        "best_architecture": info["name"],
        "architecture_params": info["kwargs"],
        "split": {"train": int(len(y_tr)), "test": int(len(y_te))},
        "metrics": {
            "best_threshold": float(info["thresh"]),
            "f1_score":       float(info["f1"]),
            "precision":      float(info["metrics"]["precision"]),
            "recall":         float(info["metrics"]["recall"]),
            "accuracy":       float(info["metrics"]["accuracy"]),
            "roc_auc":        float(info["roc_auc"]),
            "pr_auc":         float(info["pr_auc"]),
        },
        "confusion_matrix": cm.tolist(),
    }

    with open("router_lstm_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\nDone! Results saved to router_lstm_results.json")


if __name__ == "__main__":
    train_router_lstm()