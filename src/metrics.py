"""
Evaluation metrics module for Stage 1.
Computes ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier score, log loss, and ECE.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

def precision_at_k(y_true, y_prob, k=10):
    """Computes Precision@k."""
    idx = np.argsort(y_prob)[::-1][:k]
    return np.mean(y_true[idx])

def recall_at_k(y_true, y_prob, k=10):
    """Computes Recall@k."""
    idx = np.argsort(y_prob)[::-1][:k]
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    return np.sum(y_true[idx]) / total_pos

def expected_calibration_error(y_true, y_prob, n_bins=5):
    """Computes Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == 1.0)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def evaluate_predictions(y_true, y_prob):
    """
    Evaluates probability predictions against true binary labels
    across all hackathon-specified metrics.
    """
    # Clip probabilities for stable log loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'precision@10': float(precision_at_k(y_true, y_prob, k=10)),
        'recall@10': float(recall_at_k(y_true, y_prob, k=10)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob_clipped)),
        'ece': float(expected_calibration_error(y_true, y_prob))
    }
    return metrics
