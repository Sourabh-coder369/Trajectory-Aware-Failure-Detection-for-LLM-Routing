# Orange Problem Architecture

This folder contains the current routing and evaluation stack for GSM8K using:

- Qwen hidden-state trajectory features for failure prediction
- Classical ML routers (Logistic Regression, Random Forest, XGBoost)
- Feature-engineering and feature-selection experiments

## End-to-End System

```text
GSM8K question
       |
       +--> Qwen2-0.5B-Instruct forward pass
       |       |
       |       +--> Layer trajectories -> cosine_sims (23) + final_entropy (1)
       |       +--> Stored in trajectory_data_f32.jsonl
       |
       +--> Router training data
                       |
                       +--> Label creation (correct vs wrong)
                       |       - Parse numeric prediction from generated_text
                       |       - Parse numeric ground truth from #### <answer>
                       |       - Correct if abs(pred - truth) < 1e-4
                       |
                       +--> Models
                                           - Baseline logistic: train_router.py
                                           - Balanced logistic: train_router_balanced.py
                                           - RF baseline: train_router_rf.py
                                           - XGBoost baseline: train_router_xgb.py
                                           - Engineered features + logistic: train_router_features.py
                                           - Engineered features + XGBoost: train_router_features_xgb.py
                                           - Feature ranking: feature_selection.py
                                           - Hyperparameter tuning: tune_router_features.py
```

## Dataset and Class Balance

Current trajectory dataset snapshot:

- Total rows: 7473
- Correct answers: 2715
- Incorrect answers: 4758
- Failure rate: 63.67%

Because failure is the positive class and dominates, this folder includes both:

- class-weighted approaches
- explicit balanced training via undersampling (`train_router_balanced.py`)

## Main Files

- `trajectory_extractor.py`: Builds trajectory dataset from Qwen hidden states
- `trajectory_data_f32.jsonl`: Feature + text records used for router training
- `train_router.py`: Baseline logistic router on raw 24 features
- `train_router_balanced.py`: Logistic router trained on balanced train split
- `feature_selection.py`: Engineered feature extraction and ranking
- `train_router_features.py`: Logistic on selected engineered features
- `train_router_features_xgb.py`: XGBoost on selected engineered features
- `tune_router_features.py`: Hyperparameter search and CV metrics

## Typical Commands

Train balanced logistic router:

```powershell
Set-Location c:/vs/Projects/Internal_convergence_router/orange_problem
c:/vs/Projects/Internal_convergence_router/.venv/Scripts/python.exe train_router_balanced.py
```
