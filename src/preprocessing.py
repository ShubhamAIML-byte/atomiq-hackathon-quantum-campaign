"""
Preprocessing pipeline for Hybrid Quantum-Classical Term Deposit Campaign.
Strictly ensures zero data leakage by fitting all transformers on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

NUMERICAL_COLS = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Zero-leakage preprocessor that fits on training data and transforms both
    train and test splits consistently.
    """
    def __init__(self, n_quantum_features=6):
        self.n_quantum_features = n_quantum_features
        self.column_transformer = None
        self.pca = None
        self.quantum_scaler = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X_df = X.copy()
        
        # Build ColumnTransformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_COLS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
            ]
        )
        
        X_classical = self.column_transformer.fit_transform(X_df)
        
        # Fit PCA and Quantum Scaler strictly on X_classical (from train data only)
        self.pca = PCA(n_components=self.n_quantum_features, random_state=42)
        X_pca = self.pca.fit_transform(X_classical)
        
        # Scale to [0, pi] for quantum angle encoding
        self.quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.quantum_scaler.fit(X_pca)
        
        return self

    def transform(self, X):
        X_df = X.copy()
        X_classical = self.column_transformer.transform(X_df)
        return X_classical

    def transform_quantum(self, X):
        X_classical = self.transform(X)
        X_pca = self.pca.transform(X_classical)
        X_quantum = self.quantum_scaler.transform(X_pca)
        return X_quantum

def prepare_data(df_train, df_test):
    """
    Prepares training and test matrices for both classical and quantum pipelines.
    """
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X_train_raw = df_train[feature_cols]
    y_train = (df_train['y'] == 'yes').astype(int).values
    
    X_test_raw = df_test[feature_cols]
    
    preprocessor = DataPreprocessor(n_quantum_features=6)
    preprocessor.fit(X_train_raw, y_train)
    
    X_train_classical = preprocessor.transform(X_train_raw)
    X_test_classical = preprocessor.transform(X_test_raw)
    
    X_train_quantum = preprocessor.transform_quantum(X_train_raw)
    X_test_quantum = preprocessor.transform_quantum(X_test_raw)
    
    return {
        'preprocessor': preprocessor,
        'X_train_classical': X_train_classical,
        'X_test_classical': X_test_classical,
        'X_train_quantum': X_train_quantum,
        'X_test_quantum': X_test_quantum,
        'y_train': y_train
    }

if __name__ == '__main__':
    from data_loader import load_all_data
    df_train, df_test, _, _ = load_all_data()
    data = prepare_data(df_train, df_test)
    print("X_train_classical shape:", data['X_train_classical'].shape)
    print("X_test_classical shape:", data['X_test_classical'].shape)
    print("X_train_quantum shape:", data['X_train_quantum'].shape)
    print("X_test_quantum shape:", data['X_test_quantum'].shape)
    print("y_train class balance:", np.bincount(data['y_train']))
