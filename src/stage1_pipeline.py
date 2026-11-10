"""
Stage 1: Subscription Prediction Pipeline.
Trains classical and quantum/hybrid classifiers on 300 training samples,
predicts continuous probabilities for all 100 test samples, ranks candidates,
and outputs the official Stage 1 result file.
"""

import os
import pandas as pd
import numpy as np

from src.data_loader import load_all_data
from src.preprocessing import prepare_data
from src.classical_models import evaluate_classical_cv
from src.quantum_classifier import QuantumKernelClassifier, HybridQuantumClassicalClassifier
from src.metrics import evaluate_predictions

def run_stage1(output_dir='results', random_state=42):
    """
    Executes Stage 1 end-to-end.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STAGE 1: SUBSCRIPTION PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    df_train, df_test, df_s2, df_constraints = load_all_data()
    print(f"Loaded: Train = {df_train.shape}, Test = {df_test.shape}")
    
    # 2. Preprocess data (fit strictly on train)
    data = prepare_data(df_train, df_test)
    X_tr_c, X_te_c = data['X_train_classical'], data['X_test_classical']
    X_tr_q, X_te_q = data['X_train_quantum'], data['X_test_quantum']
    y_tr = data['y_train']
    
    # 3. Benchmark Classical Baselines via 5-Fold Stratified CV
    print("\n[Stage 1.1] Benchmarking Classical Baselines (Stratified 5-Fold CV)...")
    cv_classical, _ = evaluate_classical_cv(X_tr_c, y_tr, n_splits=5)
    for model_name, m in cv_classical.items():
        print(f"  {model_name:20s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier_score']:.4f}")
        
    # 4. Train Quantum Kernel Classifier & Hybrid Ensemble
    print("\n[Stage 1.2] Training Quantum Kernel Classifier (QSVC) & Hybrid Ensemble...")
    hybrid_model = HybridQuantumClassicalClassifier(quantum_weight=0.3, num_qubits=6, random_state=random_state)
    hybrid_model.fit(X_tr_c, X_tr_q, y_tr)
    
    # 5. Predict continuous probabilities on Test Customers
    print("\n[Stage 1.3] Predicting subscription probabilities for 100 test customers...")
    test_probs = hybrid_model.predict_proba(X_te_c, X_te_q)[:, 1]
    
    # Build Stage 1 test results DataFrame
    df_stage1 = pd.DataFrame({
        'customer_id': df_test['customer_id'],
        'predicted_probability': test_probs
    })
    
    # 6. Rank customers: descending probability, tie-breaker: customer_id ascending
    # We sort by predicted_probability descending, customer_id ascending
    df_stage1 = df_stage1.sort_values(
        by=['predicted_probability', 'customer_id'],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    df_stage1['rank'] = range(1, len(df_stage1) + 1)
    df_stage1['top_10_flag'] = (df_stage1['rank'] <= 10).astype(int)
    
    # Sort back by customer_id or keep by rank (submission format asks for key customer_id)
    # Submission guidelines: "for all 100 test customers, the customer_id, predicted_probability, rank, and a top-10 flag"
    # Keep ranked order or customer_id order with rank column
    output_path = os.path.join(output_dir, 'stage1_result.csv')
    df_stage1.to_csv(output_path, index=False)
    print(f"Stage 1 result saved to: {output_path}")
    
    # Top 10 Candidates summary
    top_10 = df_stage1[df_stage1['top_10_flag'] == 1].copy()
    print("\nTop 10 Stage 2 Candidates:")
    print("-" * 55)
    print(f"{'Rank':<6}{'Customer ID':<15}{'Predicted Prob':<18}{'Top 10 Flag':<12}")
    print("-" * 55)
    for _, row in top_10.iterrows():
        print(f"{int(row['rank']):<6}{row['customer_id']:<15}{row['predicted_probability']:<18.4f}{int(row['top_10_flag']):<12}")
    print("-" * 55)
    
    return df_stage1, top_10

if __name__ == '__main__':
    df_stage1, top_10 = run_stage1()
