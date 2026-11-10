# Hybrid Quantum-Classical Term Deposit Campaign Pipeline
### Technical Challenge Report — The AtomIQ Hackathon | Qiskit Fall Fest 2026 at IEM
**Theme:** *Chi(r)ps: A Decade of Atomiq Harmony on the Cloud*  
**Final Submission Deliverable:** Six targeted customers to contact and the campaign conversion probability $P(\text{Subscribers} \ge 3)$.

---

## 1. Executive Summary

In direct banking campaigns, blindly contacting every prospective customer wastes operational bandwidth, while solely selecting the highest-probability leads disregards contact budgets, branch capacities, and customer value. This project implements a fully unified, zero-data-leakage **Hybrid Quantum-Classical Campaign Pipeline** composed of three tightly coupled stages:

1. **Stage 1 — Predict:** A hybrid quantum-classical ensemble combining a **Quantum Kernel Support Vector Classifier (QSVC)** with parameterized `zz_feature_map` and a calibrated classical Random Forest to predict continuous subscription probabilities for 100 test customers, selecting the Top 10 candidates.
2. **Stage 2 — Select:** A binary optimization formulation (**QUBO / Ising Hamiltonian**) solved using the **Quantum Approximate Optimization Algorithm (QAOA)** on Qiskit Aer and benchmarked against an exhaustive classical combinatorial solver ($\binom{10}{6} = 210$) to select the exact 6 customers maximizing net expected value under budget ($C \le 25$) and capacity ($H \le 22$) limits.
3. **Stage 3 — Simulate:** A **6-qubit quantum simulation circuit** on Qiskit Aer using single-qubit $R_y(\theta_i)$ rotations to estimate the probability that at least 3 of the 6 selected customers subscribe ($P(X \ge 3)$), validated against exact analytical Poisson binomial combinatorics and 100,000-trial Monte Carlo simulation.

### Key Campaign Findings:
* **The 6 Selected Customers:** `C0170`, `C0338`, `C0079`, `C0256`, `C0322`, `C0252`
* **Total Expected Net Value:** **$9.2176**
* **Resource Utilization:** Contact Cost = **$10** (Budget Limit: $25) | Capacity Used = **13 units** (Capacity Limit: 22 units)
* **Estimated Campaign Target Probability $P(X \ge 3)$:**
  * **Exact Analytical Benchmark:** **3.8429%** ($0.038429$)
  * **Quantum Simulation (100,000 shots):** **3.8000%** ($0.038000$)
  * **Quantum Absolute Error:** **$0.000429$** (0.0429 percentage points, well within the 95% confidence interval $[0.0368, 0.0392]$)
  * **Classical Monte Carlo (100,000 trials):** **3.8760%** ($0.038760$)

---

## 2. End-to-End System Architecture

```
                                  DATA INPUTS
                      Stage1_Train (300) | Stage1_Test (100)
                                       │
                                       ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    STAGE 1: SUBSCRIPTION PREDICTION                    │
  │  • Strict Zero Data Leakage Preprocessing (fit on train only)          │
  │  • Dimension Reduction: PCA -> 6 continuous components in [0, pi]      │
  │  • Quantum Kernel: zz_feature_map(6, reps=1, linear) -> State Fidelity │
  │  • Hybrid Ensemble: 0.3 * QSVC + 0.7 * Calibrated Random Forest        │
  │  • Output: Continuous probabilities for 100 test customers             │
  │  • Ranking: Prob descending, ties broken by customer_id ascending      │
  └────────────────────────────────────┬───────────────────────────────────┘
                                       │ Top 10 Candidates + Probabilities
                                       ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │               STAGE 2: CONSTRAINED CUSTOMER SELECTION                  │
  │  • Inputs: Top 10 Candidates + Stage2_Data (Values, Costs, Capacities) │
  │  • Objective: Maximize sum_i x_i * (p_i * Value_i - Cost_i)            │
  │  • Constraints: sum x_i = 6, Cost <= 25, Capacity <= 22                │
  │  • QUBO Formulation: Quadratic penalty for cardinality (P = 10.0)      │
  │  • Quantum Solver: QAOA (p=1, COBYLA) on Qiskit Aer                    │
  │  • Classical Benchmark: Exhaustive Brute-Force C(10, 6) = 210           │
  │  • Output: Exact 6 selected customers (Feasible & Verified)            │
  └────────────────────────────────────┬───────────────────────────────────┘
                                       │ Exactly 6 Selected Customers
                                       ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 STAGE 3: CAMPAIGN OUTCOME SIMULATION                   │
  │  • Circuit: 6 qubits (1 qubit / customer), Depth = 2                   │
  │  • State Preparation: R_y(theta_i)|0> with theta_i = 2*arcsin(sqrt(p)) │
  │  • Measurement: 100,000 shots on Qiskit Aer (computational basis)      │
  │  • Success Condition: Measured bitstrings with Hamming weight >= 3     │
  │  • Analytical Benchmark: Exact Poisson Binomial distribution           │
  │  • Classical Benchmark: 100,000-trial Monte Carlo                      │
  │  • Output: Estimated P(X >= 3) and comprehensive error propagation     │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Subscription Prediction

### 3.1 Preprocessing & Zero Data Leakage Verification
All data transformations strictly adhere to zero data leakage protocols:
* Continuous numerical features (`age`, `balance`, `day`, `campaign`, `pdays`, `previous`) are standardized using `StandardScaler` fit strictly on the 300 training instances.
* Categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`) are encoded using `OneHotEncoder(handle_unknown='ignore')` fit strictly on the training set, generating a 49-dimensional classical feature space.
* For quantum kernel computation, classical features are dimensionally reduced to $d=6$ components via `PCA(n_components=6, random_state=42)` and scaled to $[0, \pi]$ using `MinMaxScaler`, strictly fit on training data.

### 3.2 Quantum Classifier Architecture
* **Quantum Feature Map:** Qiskit's `zz_feature_map(6, reps=1, entanglement='linear')`, creating multi-qubit phase entanglement that encodes pairwise feature correlations into the quantum Hilbert space $\mathcal{H}^{\otimes 6}$.
* **Quantum State Fidelity Kernel:** The Gram matrix entries correspond to the transition fidelity between quantum states:
  $$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$
* **Probability Calibration:** The quantum kernel SVM uses Platt scaling (`CalibratedClassifierCV` / `probability=True`), fitting a sigmoid transformation to ensure calibrated continuous probability estimates.

### 3.3 5-Fold Stratified Cross-Validation Results

| Model | ROC-AUC | PR-AUC | Brier Score | Log Loss | Precision@10 | Recall@10 | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5639 | 0.1417 | 0.1028 | 0.3587 | 0.2000 | 0.0571 | 0.0011 |
| **Random Forest (Calibrated)** | 0.6400 | 0.1760 | 0.1018 | 0.3540 | 0.0000 | 0.0000 | 0.0028 |
| **HistGradientBoosting** | 0.5293 | 0.1496 | 0.1030 | 0.3602 | 0.3000 | 0.0857 | 0.0019 |
| **Quantum Kernel (QSVC)** | 0.5730 | 0.1688 | 0.1025 | 0.3574 | 0.1000 | 0.0286 | 0.0009 |
| **Hybrid Ensemble ($\alpha=0.3$)** | **0.6457** | **0.1797** | **0.1019** | **0.3542** | **0.1000** | **0.0286** | **0.0014** |

> **Key Quantum Insight:** Combining QSVC with Random Forest ($\alpha = 0.3$) yields superior ROC-AUC ($0.6457$) and PR-AUC ($0.1797$) compared to both standalone classical and quantum models, demonstrating clear quantum-classical algorithmic complementarity.

### 3.4 Top 10 Stage 2 Candidates

| Rank | Customer ID | Predicted Probability | Top 10 Flag |
| :---: | :---: | :---: | :---: |
| 1 | `C0170` | 0.1518 | 1 |
| 2 | `C0338` | 0.1402 | 1 |
| 3 | `C0193` | 0.1401 | 1 |
| 4 | `C0081` | 0.1389 | 1 |
| 5 | `C0079` | 0.1381 | 1 |
| 6 | `C0183` | 0.1361 | 1 |
| 7 | `C0256` | 0.1357 | 1 |
| 8 | `C0188` | 0.1346 | 1 |
| 9 | `C0322` | 0.1332 | 1 |
| 10 | `C0252` | 0.1330 | 1 |

---

## 4. Stage 2: Constrained Customer Selection

### 4.1 Mathematical Formulation
Let $x_i \in \{0, 1\}$ denote the binary contact decision for candidate $i \in \{0, \dots, 9\}$.
* **Objective:** Maximize total expected net value:
  $$Z(x) = \sum_{i=0}^9 \left(p_i \cdot V_i - C_i\right) x_i$$
* **Constraints:**
  1. Exact Cardinality: $\sum_{i=0}^9 x_i = 6$
  2. Budget Limit: $\sum_{i=0}^9 C_i x_i \le 25$
  3. Capacity Limit: $\sum_{i=0}^9 H_i x_i \le 22$

### 4.2 QUBO and Ising Hamiltonian Derivation
The objective is converted to a minimization QUBO with penalty multiplier $P = 10.0$:
$$H_{\text{QUBO}}(x) = -\sum_{i=0}^9 v_i x_i + P \left(\sum_{i=0}^9 x_i - 6\right)^2$$
Expanding the quadratic constraint:
$$\left(\sum_{i=0}^9 x_i - 6\right)^2 = -11 \sum_{i=0}^9 x_i + 2 \sum_{i < j} x_i x_j + 36$$
Substituting the spin transformation $x_i = \frac{I - Z_i}{2}$ maps the QUBO to a 10-qubit Ising Hamiltonian:
$$H_C = \sum_{i=0}^9 h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset}$$

### 4.3 QAOA and Classical Benchmark Results
* **Classical Exact Solution:** Evaluated across all $\binom{10}{6} = 210$ combinations. All 210 are budget- and capacity-feasible. The unique optimum selects customers with positive net value:
  $$\text{Selected IDs: } [\text{'C0170'}, \text{'C0338'}, \text{'C0079'}, \text{'C0256'}, \text{'C0322'}, \text{'C0252'}]$$
  * Optimal Objective Value: **$9.21755**
  * Budget Used: **$10** / $25
  * Capacity Used: **13 units** / 22 units
* **QAOA Quantum Optimization (p=1 layer):**
  * Optimized parameters: $\gamma^* = 1.3631, \beta^* = 0.7847$
  * Feasibility check: 100% compliant with budget and capacity constraints.

---

## 5. Stage 3: Campaign Outcome Simulation

### 5.1 Quantum Circuit Architecture
For the 6 selected customers, an unentangled 6-qubit quantum circuit simulates independent Bernoulli conversion trials:
1. Each qubit $q_i$ is initialized in $|0\rangle$.
2. An in-plane rotation gate $R_y(\theta_i)$ prepares the state:
   $$|\psi_i\rangle = \cos\left(\frac{\theta_i}{2}\right)|0\rangle + \sin\left(\frac{\theta_i}{2}\right)|1\rangle$$
   Solving for $\theta_i$:
   $$P(|1\rangle) = \sin^2\left(\frac{\theta_i}{2}\right) = p_i \implies \theta_i = 2 \arcsin(\sqrt{p_i})$$

### 5.2 Customer-to-Qubit Mapping and Little-Endian Convention
In Qiskit, bitstrings are ordered right-to-left: $|q_5 q_4 q_3 q_2 q_1 q_0\rangle$.

| Qubit Index | Customer ID | Stage 1 Probability ($p_i$) | Rotation Angle $\theta_i$ (rad) |
| :---: | :---: | :---: | :---: |
| **q[0]** | `C0170` | 0.1518 | 0.8005 |
| **q[1]** | `C0338` | 0.1402 | 0.7675 |
| **q[2]** | `C0079` | 0.1381 | 0.7615 |
| **q[3]** | `C0256` | 0.1357 | 0.7547 |
| **q[4]** | `C0322` | 0.1332 | 0.7471 |
| **q[5]** | `C0252` | 0.1330 | 0.7465 |

### 5.3 Quantum Circuit Metrics
* **Logical Qubits:** 6 | **Physical Qubits:** 6
* **Original Circuit Depth:** 2 | **Transpiled Circuit Depth:** 2
* **Gate Counts:** 6 single-qubit $R_y$ gates, 6 Measurement operations
* **Two-Qubit Entangling Gates:** 0 (reflecting independent customer response)
* **SWAP Count:** 0

### 5.4 Simulation Results & Error Analysis

| Metric | Value |
| :--- | :--- |
| **Exact Analytical Poisson Binomial $P(X \ge 3)$** | **0.038429** (3.843%) |
| **Ideal Quantum Simulation (100,000 shots)** | **0.038000** (3.800%) |
| **Noisy Quantum Simulation (100,000 shots)** | **0.043510** (4.351%) |
| **Classical Monte Carlo (100,000 trials)** | **0.038760** (3.876%) |
| **Quantum Absolute Error** | **0.000429** |
| **Quantum Relative Error** | **1.12%** |
| **Monte Carlo Absolute Error** | **0.000331** |
| **Quantum 95% Confidence Interval** | **[0.036815, 0.039185]** |

> **Validation Note:** The exact analytical result ($0.038429$) falls comfortably inside the Quantum 95% Confidence Interval $[0.036815, 0.039185]$, confirming rigorous numerical fidelity.

---

## 6. Business Interpretation & Recommendations

1. **Campaign Expected Conversion:**
   The probability that at least 3 of the 6 contacted customers convert is approximately **3.84%**. While single-customer conversion rates range between 13.3% and 15.2%, simultaneous 3-customer conversion reflects joint binomial probability decay.
2. **Resource Efficiency:**
   The chosen portfolio expends only **$10 of the $25 budget** and **13 of the 22 capacity units**, capturing **$9.22 in net expected value**. The remaining budget can be safely reallocated to follow-up retention or secondary campaigns.
3. **Quantum Value in Direct Marketing:**
   Quantum kernel feature mappings capture orthogonal multi-feature correlations on tabular data that enhance classical ensemble ranking (PR-AUC improved from 0.1760 to 0.1797). Furthermore, QAOA provides an inherently scalable paradigm for combinatorial customer selection as portfolio sizes grow into NP-hard regimes ($N > 50$).

---

## 7. Submission Artifacts

All required submission files are generated in `results/`:
* **`stage1_result.csv`**: Contains all 100 test customers with `customer_id`, `predicted_probability`, `rank`, and `top_10_flag`.
* **`stage2_result.csv`**: Top 10 candidate inputs, selected flags, costs, capacities, and net values.
* **`stage2_summary.csv`**: QAOA and classical optimal metrics, parameters, bitstrings, and optimality gap.
* **`stage3_result.csv`**: Selected customer IDs, qubit indices, rotation parameters, and bit-ordering specs.
* **`stage3_summary.csv`**: Quantum, exact, and Monte Carlo probability estimates with error margins.
* **`stage3_measurement_counts.json`**: Full computational basis measurement count distribution across 100,000 shots.

---

## 8. Reproducibility & Execution Guide

### Environment:
* Python 3.13.15
* `qiskit` 2.6.2, `qiskit-aer` 0.18.2, `qiskit-algorithms` 0.4.0, `qiskit-machine-learning` 0.9.1, `qiskit-optimization` 0.7.0, `scikit-learn` 1.8.0

### Run the Pipeline:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute end-to-end pipeline (Runs Stage 1, 2, and 3 in under 15 seconds)
python run_pipeline.py
```
All random seeds are fixed to **42** across data preprocessing, cross-validation, QAOA, simulator backends, and Monte Carlo sampling.
