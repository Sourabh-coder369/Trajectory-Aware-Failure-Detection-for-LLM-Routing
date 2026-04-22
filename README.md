# Trajectory-Aware-LLM-Routing-For-Failure-Detection

## Project Overview

This project builds a **failure-prediction router** for a small language model (Qwen2-0.5B-Instruct) on the GSM8K math benchmark. The core idea: instead of always escalating a question to a larger, more expensive model, we extract internal signals from the small model's forward pass and train a lightweight classifier to predict *before seeing the final answer* whether the small model is likely to get the question wrong. If the router predicts failure, the question is routed to a stronger model.

### Why This Works

During a forward pass, each transformer layer produces a hidden state. Two signals are extracted:

- **Cosine similarities** between adjacent layer hidden states (23 values for a 24-layer model) — measures how much the representation changes layer-to-layer. Stable, converging representations tend to correlate with correct answers.
- **Final token entropy** (1 value) — measures how confident the model is in its next-token prediction at the last step. High entropy signals uncertainty.

These 24 features are cheap to compute and available before decoding the full answer, making them useful for routing decisions.

### Task Setup

- **Dataset**: GSM8K (grade school math), 7473 samples
- **Small model**: Qwen2-0.5B-Instruct
- **Positive class**: Failure (label 1 = wrong answer)
- **Failure rate**: 63.67% — the dataset is imbalanced, failure dominates
- **Label definition**: `abs(predicted_number - ground_truth_number) < 1e-4`

---

## Model Results

All models are trained on an 80/20 train/test split (stratified). Threshold is swept from 0.1 to 0.95 and the value maximizing F1 is reported.

| Model | Threshold | F1 | Precision | Recall | Accuracy | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.40 | 0.777 | 0.664 | 0.937 | 0.662 | — | — |
| Random Forest | 0.35 | 0.792 | 0.663 | 0.983 | 0.675 | — | — |
| XGBoost | 0.25 | 0.801 | 0.693 | 0.948 | 0.703 | — | — |
| MLP (Shallow 64) | 0.25 | 0.800 | 0.709 | 0.919 | 0.708 | 0.748 | 0.822 |

### Key Observations

- **All models achieve high recall (>0.91)** — they rarely miss a failing case, which is the primary goal for a safety-oriented router.
- **Precision is moderate (0.66–0.71)** — some correct answers get unnecessarily escalated, but this is the conservative trade-off.
- **XGBoost and MLP are the strongest**, both reaching F1 ≈ 0.80.
- **Random Forest achieves the highest recall (0.983)** at the cost of lower precision — almost no failures are missed.
- **Logistic Regression is a strong baseline** given its simplicity, only ~2 F1 points behind the best models.

### MLP Architecture Comparison

Three architectures were evaluated; the shallow network won:

| Architecture | Result |
|---|---|
| Shallow (64) | **Best — F1 0.800, ROC AUC 0.748, PR AUC 0.822** |
| Medium (128-64) | Marginally worse |
| Deep (256-128-64) | No additional gain |

The shallow architecture winning suggests the 24-feature input does not benefit from deep representations.

### Feature Importance (XGBoost)

The most important cosine similarity is at **layer 5** (importance ≈ 0.096), suggesting mid-network representation shifts are the strongest signal. Final token entropy has relatively low standalone importance (0.045) but contributes in combination with cosine features.

---

## Tracked Files

### Training Scripts

| File | Model | Handles imbalance via |
|---|---|---|
| `train_router.py` | Logistic Regression | `class_weight='balanced'` |
| `train_router_rf.py` | Random Forest | `class_weight='balanced'` |
| `train_router_xgb.py` | XGBoost | `scale_pos_weight` |
| `train_router_mlp.py` | MLP | Undersampling (balanced train set) |

### Result Files

| File | Contents |
|---|---|
| `results/router_results.json` | LR threshold, F1, precision, recall, layer coefficients |
| `results/router_rf_results.json` | RF threshold, F1, feature importances per layer |
| `results/router_xgb_results.json` | XGB threshold, F1, feature importances per layer |
| `results/router_mlp_results.json` | MLP architecture comparison, full metrics, confusion matrix |

> **Note**: Input data (`data/`) is gitignored. Run the trajectory extractor to regenerate `data/trajectory_data_f32.jsonl` before training.

---

## Running the Models

```powershell
Set-Location c:/vs/Projects/Internal_convergence_router/orange_problem
$python = "c:/vs/Projects/Internal_convergence_router/.venv/Scripts/python.exe"

& $python train_router.py        # Logistic Regression
& $python train_router_rf.py     # Random Forest
& $python train_router_xgb.py    # XGBoost
& $python train_router_mlp.py    # MLP
```