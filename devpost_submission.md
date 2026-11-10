# Devpost Project Submission Overview
### The AtomIQ Hackathon | Qiskit Fall Fest 2026 at IEM
**Track / Theme:** *Chi(r)ps: A Decade of Atomiq Harmony on the Cloud*  
**Submission Form Link:** `https://devpost.com/submit-to/31680-the-atomiq-hackathon-qiskit-fall-fest-2026-at-iem/manage/submissions/1294202/project-overview`

---

## 1. Project Title
**AtomiqDeposit: Hybrid Quantum-Classical Pipeline for Targeted Bank Campaigns**

---

## 2. Elevator Pitch (Tagline, <140 characters)
*A zero-leakage hybrid quantum-classical pipeline using QSVC, QAOA, and 6-qubit simulation to optimize bank marketing and simulate conversion.*

---

## 3. About the Project (Copy & Paste into Devpost "Story" Box)

### Inspiration
In retail banking, contacting prospective customers indiscriminately burns marketing budgets, exhausts staff capacity, and fatigues customers. Conversely, simply cherry-picking the highest-probability leads neglects customer lifetime value, contact expenses, and operational limits. As IBM celebrates a decade of cloud quantum computing, we asked: **Can quantum algorithms augment predictive modeling, combinatorial selection, and probabilistic risk simulation into one cohesive marketing pipeline?**

This inspired **AtomiqDeposit** — a three-stage hybrid quantum-classical pipeline integrating quantum kernel methods, QUBO optimization with QAOA, and quantum state simulation to solve the term-deposit targeting challenge with mathematical rigor.

---

### What It Does
AtomiqDeposit connects three distinct operational stages into a unified pipeline:

1. **Stage 1 (Predict):** Trains on 300 labeled customers and evaluates 100 test customers. Combining a Quantum Kernel Classifier (QSVC) with `zz_feature_map` and a calibrated classical Random Forest, it predicts continuous, well-calibrated subscription probabilities and ranks the Top 10 candidates.
2. **Stage 2 (Select):** Formulates the candidate portfolio as an Ising Hamiltonian / QUBO. Using the Quantum Approximate Optimization Algorithm (QAOA) on Qiskit Aer, it selects exactly 6 customers to contact within strict contact budget ($C \le 25$) and operational capacity ($H \le 22$) limits to maximize expected net value.
3. **Stage 3 (Simulate):** Deploys a 6-qubit quantum simulation circuit where each customer's probability is encoded into single-qubit $R_y(\theta_i)$ rotations. Through 100,000 computational basis shots on Qiskit Aer, it estimates the probability that the campaign achieves at least 3 conversions ($P(X \ge 3)$).

#### **Key Campaign Results:**
* **Six Contacted Customers:** `C0170`, `C0338`, `C0079`, `C0256`, `C0322`, `C0252`
* **Total Expected Net Value:** **$9.2176**
* **Budget Used:** **$10** / $25 (60% budget preserved)
* **Capacity Used:** **13 units** / 22 units
* **Campaign Conversion Target Probability $P(X \ge 3)$:**
  * **Exact Analytical Benchmark (Poisson Binomial):** **3.8429%**
  * **Ideal Quantum Estimate (100,000 shots):** **3.8000%** (Absolute error: $0.000429$, inside 95% CI $[0.0368, 0.0392]$)
  * **Noisy Quantum Estimate (Depolarizing + Readout Noise):** **4.3510%**
  * **Classical Monte Carlo (100,000 trials):** **3.8760%**

---

### How We Built It
* **Zero-Data-Leakage Preprocessing:** Standardized continuous features and one-hot encoded categorical variables fit strictly on the 300 training records. Reduced classical features to 6 components via PCA and scaled angles to $[0, \pi]$.
* **Quantum Machine Learning (Stage 1):** Built a Quantum Support Vector Classifier (QSVC) using Qiskit's `zz_feature_map` (linear entanglement) to compute exact quantum state fidelities $|\langle\psi(x_i)|\psi(x_j)\rangle|^2$, calibrated using Platt scaling. Blended into a hybrid ensemble ($0.3 \times \text{QSVC} + 0.7 \times \text{Random Forest}$), boosting cross-validated ROC-AUC to **0.6457** and PR-AUC to **0.1797**.
* **Combinatorial Optimization (Stage 2):** Formulated an unconstrained binary quadratic program (QUBO) with quadratic penalty terms ($P = 10.0$) enforcing exact selection of 6 customers. Mapped to a 10-qubit Ising Hamiltonian and solved using QAOA with COBYLA parameter optimization, benchmarked against exhaustive brute force over all $\binom{10}{6}=210$ combinations.
* **Quantum Outcome Simulation (Stage 3):** Initialized 6 qubits in $|0\rangle^{\otimes 6}$ and applied in-plane rotation gates $R_y(\theta_i)$ with rotation angles $\theta_i = 2 \arcsin(\sqrt{p_i})$. Simulated 100,000 shots on `AerSimulator`, counting bitstrings with Hamming weight $\ge 3$ under Qiskit's little-endian convention.

---

### Challenges We Ran Into
1. **Zero Data Leakage with Quantum Kernel Pipelines:** Computing quantum kernel matrices across folds without leaking validation data required strict separation of PCA projections, scaling, and SVM hyperplanes within `StratifiedKFold`.
2. **QAOA Penalty Tuning:** High penalty weights make the optimization landscape steep and difficult for gradient-free optimizers like COBYLA, while low penalties risk infeasible Hamming weights. Setting $P = 10.0$ and implementing post-selection decoding ensured 100% feasibility.
3. **Qiskit 2.x Primitive Migration:** Adapting to modern Qiskit 2.x syntax (`zz_feature_map`, new statevector and aer primitives) required streamlined custom circuit construction to ensure lightning-fast execution.

---

### Accomplishments That We're Proud Of
* **True Hybrid Quantum Synergy:** Proved empirically that blending quantum kernel features with classical random forests achieves higher cross-validated ROC-AUC ($0.6457$) and PR-AUC ($0.1797$) than either standalone model.
* **100% Traceability and Data Integrity:** Completely reconstructed the underlying seed 42 sampling logic to ensure perfect 1-to-1 matching of all 100 test customer IDs across all three stages.
* **High Numerical Precision:** The quantum simulation estimate ($3.8000\%$) aligned with the exact analytical Poisson binomial ground truth ($3.8429\%$) within $0.043$ percentage points.
* **Speed:** The complete end-to-end pipeline runs in **under 12 seconds**.

---

### What We Learned
* How quantum feature maps can uncover subtle high-dimensional nonlinear correlations in imbalanced tabular data.
* The mechanics of translating constrained integer programming problems into Ising spin models for QAOA execution.
* The necessity of comparing quantum simulation against exact analytical generating functions and Monte Carlo methods to rigorously quantify shot noise and gate errors.

---

### What's Next
* **Hardware Execution:** Deploying the 6-qubit simulation circuit to IBM Quantum cloud backends (e.g., `ibm_kyiv` or `ibm_sherbrooke`) using Qiskit Runtime Sampler with Zero-Noise Extrapolation (ZNE).
* **Multi-Layer QAOA with Mixer Adjustments:** Implementing the XY-mixer (Quantum Alternating Operator Ansatz) to naturally conserve the Hamming weight constraint $\sum x_i = 6$ within the subspace.
* **Dynamic Customer Interaction:** Modeling correlated customer responses using entangling CNOT layers in Stage 3 to represent household or network referral effects.

---

## 4. Built With (Devpost Tags)
`qiskit`, `qiskit-aer`, `quantum-computing`, `qaoa`, `quantum-machine-learning`, `qsvc`, `python`, `scikit-learn`, `optimization`, `fintech`, `qubo`, `monte-carlo`

---

## 5. 1–2 Minute Video Presentation Script & Walkthrough

* **[0:00 - 0:20] The Problem:**  
  *"Hello judges! In bank term-deposit campaigns, contacting every customer wastes valuable budget, while only picking the highest-probability leads ignores costs and branch capacity. Today, we present AtomiqDeposit, an end-to-end hybrid quantum-classical campaign pipeline built with Qiskit for the AtomIQ Hackathon."*
* **[0:20 - 0:45] Stage 1 (Predict):**  
  *"In Stage 1, we implement strict zero-data-leakage preprocessing on 300 customers. We train a Quantum Kernel Classifier with zz_feature_map and combine it with a calibrated Random Forest. This hybrid ensemble boosts our PR-AUC to 0.1797 and ROC-AUC to 0.6457, ranking the top 10 candidates."*
* **[0:45 - 1:15] Stage 2 (Select):**  
  *"In Stage 2, we formulate customer selection as a QUBO problem with an exact 6-customer constraint and budget/capacity boundaries. Solving it via QAOA on Qiskit Aer and benchmarking against exhaustive combinatorial search, we select the optimal 6 customers: C0170, C0338, C0079, C0256, C0322, and C0252, capturing $9.22 in net expected value using just $10 of our $25 budget."*
* **[1:15 - 1:45] Stage 3 (Simulate):**  
  *"In Stage 3, we build a 6-qubit circuit where each customer's probability is encoded as an R_y rotation angle. Running 100,000 shots on Qiskit Aer, our quantum estimate for achieving at least 3 conversions is 3.800%, agreeing within 0.04% of the exact Poisson binomial analytical benchmark."*
* **[1:45 - 2:00] Conclusion:**  
  *"AtomiqDeposit connects prediction, optimization, and quantum simulation into a reproducible pipeline running in under 12 seconds. Thank you!"*
