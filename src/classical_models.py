"""
Classical probabilistic classifier baselines with probability calibration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from src.metrics import evaluate_predictions

def get_classical_models():
    """Returns a dictionary of classical baseline models."""
    models = {
        'LogisticRegression': LogisticRegression(
            C=0.5, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=150, max_depth=6, class_weight='balanced', random_state=42
        ),
        'GradientBoosting': HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, class_weight='balanced', random_state=42
        )
    }
    return models

def evaluate_classical_cv(X, y, n_splits=5):
    """
    Evaluates classical models using Stratified 5-Fold Cross Validation.
    Applies Platt scaling / Sigmoid calibration.
    """
    models = get_classical_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {}
    oof_predictions = {}
    
    for name, base_model in models.items():
        oof_probs = np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
            calibrated_model.fit(X_tr, y_tr)
            
            oof_probs[val_idx] = calibrated_model.predict_proba(X_val)[:, 1]
            
        metrics = evaluate_predictions(y, oof_probs)
        cv_results[name] = metrics
        oof_predictions[name] = oof_probs
        
    return cv_results, oof_predictions

if __name__ == '__main__':
    from src.data_loader import load_all_data
    from src.preprocessing import prepare_data
    
    df_train, df_test, _, _ = load_all_data()
    data = prepare_data(df_train, df_test)
    
    results, oof = evaluate_classical_cv(data['X_train_classical'], data['y_train'])
    print("=== Classical CV Results ===")
    for model_name, m in results.items():
        print(f"\n{model_name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
