import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# Paths
model_path = 'project_files/model/catboost_model.pkl'
test_data_path = 'project_files/test_data/test_data.csv'
labels_path = 'project_files/test_data/labels.csv'

def optimize_threshold_eval(iters, depth, importance, learning_rate):
    data = pd.read_csv(test_data_path)
    labels_df = pd.read_csv(labels_path)
    true_labels = labels_df['inhospital_mortality']

    # Load model and selected features
    model, selected_features = load(model_path)
    model._feature_names_used = selected_features

    # Filter and cast categorical features
    data = data[selected_features]
    for col in data.select_dtypes(include=['object', 'category']).columns:
        data[col] = data[col].astype(str)

    # Predict probabilities
    pred_probs = model.predict_proba(data)[:, 1]

    # Global metrics (threshold-independent)
    auc_score = roc_auc_score(true_labels, pred_probs)
    prec, rec, _ = precision_recall_curve(true_labels, pred_probs)
    auprc_score = auc(rec, prec)

    def calculate_ece(probs, labels, n_bins=10):
        edges = np.linspace(0,1,n_bins+1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs > edges[i]) & (probs <= edges[i+1])
            if mask.sum() > 0:
                ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean()) / len(probs)
        return ece

    ece_score = calculate_ece(pred_probs, true_labels)

    # Scan thresholds [modified from (0, 0.1, 2000) to (0, 0.01, 3500)]
    thresholds = np.linspace(0, 0.01, 3500)
    results = []

    for threshold in thresholds:
        preds = (pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(true_labels, preds)
        accuracy = accuracy_score(true_labels, preds)
        net_benefit = (tp / len(true_labels)) - (fp / len(true_labels)) * (threshold / (1 - threshold)) if threshold < 1 else 0

        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy,
            'net_benefit': net_benefit,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'AUC': auc_score,
            'AUPRC': auprc_score,
            'ECE': ece_score
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f'Threshold optimization results for iters={iters}, depth={depth}, importance={importance}, learning_rate={learning_rate}:')
    # Show top thresholds by Net Benefit
    df_filtered = df[df['sensitivity'] >= 0.8].sort_values(by='net_benefit', ascending=False)
    print(df_filtered.to_string(index=False))

    return df_filtered.to_string(index=False)