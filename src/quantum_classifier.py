"""
Quantum and Hybrid probabilistic classifier module using Qiskit.
Implements Quantum Kernel Classifier (QSVC) with zz_feature_map and
Hybrid Quantum-Classical Ensemble.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

class QuantumKernelClassifier:
    """
    Quantum Kernel Support Vector Classifier using Qiskit zz_feature_map
    with probability calibration.
    """
    def __init__(self, num_qubits=6, reps=1, entanglement='linear', C=1.0, random_state=42):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.random_state = random_state
        self.feature_map = zz_feature_map(
            self.num_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )
        self.X_train_ = None
        self.Psi_train_ = None
        self.svc_ = None

    def _compute_statevectors(self, X):
        statevectors = []
        for x in X:
            param_dict = dict(zip(self.feature_map.parameters, x))
            bound_circ = self.feature_map.assign_parameters(param_dict)
            sv = Statevector.from_instruction(bound_circ).data
            statevectors.append(sv)
        return np.array(statevectors)

    def fit(self, X, y):
        self.X_train_ = X
        self.Psi_train_ = self._compute_statevectors(X)
        # Compute exact quantum state fidelity Gram matrix: |<psi_i | psi_j>|^2
        K_train = np.abs(self.Psi_train_ @ self.Psi_train_.conj().T) ** 2
        
        # Train SVC with Platt scaling (probability=True)
        self.svc_ = SVC(
            kernel='precomputed',
            C=self.C,
            probability=True,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.svc_.fit(K_train, y)
        return self

    def predict_proba(self, X):
        Psi_test = self._compute_statevectors(X)
        K_test = np.abs(Psi_test @ self.Psi_train_.conj().T) ** 2
        return self.svc_.predict_proba(K_test)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class HybridQuantumClassicalClassifier:
    """
    Ensemble combining Quantum Kernel Classifier and Classical Random Forest.
    """
    def __init__(self, quantum_weight=0.3, num_qubits=6, random_state=42):
        self.quantum_weight = quantum_weight
        self.num_qubits = num_qubits
        self.random_state = random_state
        self.q_model = QuantumKernelClassifier(num_qubits=num_qubits, random_state=random_state)
        from sklearn.ensemble import RandomForestClassifier
        self.c_model = CalibratedClassifierCV(
            estimator=RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=random_state),
            method='sigmoid',
            cv=3
        )

    def fit(self, X_classical, X_quantum, y):
        self.c_model.fit(X_classical, y)
        self.q_model.fit(X_quantum, y)
        return self

    def predict_proba(self, X_classical, X_quantum):
        p_c = self.c_model.predict_proba(X_classical)[:, 1]
        p_q = self.q_model.predict_proba(X_quantum)[:, 1]
        p_hybrid = self.quantum_weight * p_q + (1.0 - self.quantum_weight) * p_c
        return np.vstack([1.0 - p_hybrid, p_hybrid]).T
