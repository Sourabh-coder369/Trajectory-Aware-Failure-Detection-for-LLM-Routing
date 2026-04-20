import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_answer(text):
    """Extract the final numeric answer from text.
    For ground_truth: looks for #### pattern first (GSM8K format).
    For generated_text: extracts the last number found.
    """
    if text is None:
        return None
    
    # Try GSM8K ground truth format: #### <number>
    match = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Fallback: find the last number in the text
    numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))

def compute_correctness(entry):
    """Determine if the model's answer is correct by comparing
    the extracted numeric answer from generated_text vs ground_truth."""
    
    # If the entry already has is_correct (old format), use it
    if 'is_correct' in entry:
        return entry['is_correct']
    
    pred_val = extract_answer(entry.get('generated_text', ''))
    truth_val = extract_answer(entry.get('ground_truth', ''))
    
    if pred_val is None or truth_val is None:
        return False
    
    return abs(pred_val - truth_val) < 1e-4

def load_data(filename="data/trajectory_data_f32.jsonl"):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_features(data):
    X = []
    y = []
    for entry in data:
        # Features: [Cosine Sims ..., Final Entropy]
        # Cosine sims is a list of length N-1
        features = entry['cosine_sims'] + [entry['final_entropy']]
        X.append(features)
        
        # Label: We want to predict FAILURE.
        # So Label=1 if WRONG, Label=0 if CORRECT.
        is_correct = compute_correctness(entry)
        label = 1 if not is_correct else 0
        y.append(label)
        
    return np.array(X), np.array(y)

def clean_data(X, y):
    # Replace NaNs with 0 to enable training
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y

def train_router():
    print("Loading data...")
    raw_data = load_data()
    if not raw_data:
        print("No data found!")
        return

    print(f"Loaded {len(raw_data)} samples.")
    
    X, y = prepare_features(raw_data)
    X, y = clean_data(X, y)
    
    # Check class balance
    print(f"Failure rate: {y.mean():.2%} ({sum(y)} failures out of {len(y)})")
    
    if len(raw_data) < 5:
        print("Not enough data to split. Training on all data.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    print(f"Training Random Forest on {len(X_train)} samples...")
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Threshold Sweep
    print("\n--- Threshold Optimization ---")
    probs = model.predict_proba(X_test)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0
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
            best_metrics = {'precision': p, 'recall': r, 'accuracy': acc}

    print(f"\n✅ Best Threshold: {best_thresh:.2f}")
    print(f"📊 Best F1 Score: {best_f1:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")

    # Compute optimal confusion matrix
    y_pred_best = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix (@ best threshold):")
    print(f"  [TN (True Negs)   FP (False Pos)] -> Actual Successs")
    print(f"  [FN (False Negs)  TP (True Pos)]  -> Actual Failures")
    print(cm)

    # Feature Importance
    print("\nFeature Importance (Gini Importance):")
    importances = model.feature_importances_
    # Last one is Entropy
    print(f"Final Entropy Importance: {importances[-1]:.4f}")
    # Rest are layers
    print(f"Layer Cosine Sims Importances (First 5): {importances[:5]}")
    print(f"Layer Cosine Sims Importances (Last 5): {importances[-6:-1]}")

    # Save to file
    output_dict = {
        "metrics": {
            "best_threshold": float(best_thresh),
            "f1_score": float(best_f1),
            "precision": float(best_metrics['precision']),
            "recall": float(best_metrics['recall']),
            "accuracy": float(best_metrics['accuracy']),
        },
        "feature_importances": {
            "final_entropy_importance": float(importances[-1]),
            "cosine_sims_importances": [float(i) for i in importances[:-1]]
        }
    }
    
    with open('router_rf_results.json', 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    print("\nDone! Results and importances saved to router_rf_results.json")

if __name__ == "__main__":
    train_router()