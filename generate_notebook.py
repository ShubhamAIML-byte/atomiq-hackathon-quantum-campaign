"""
Script to generate the comprehensive, fully-functional Jupyter Notebook: hybrid_pipeline.ipynb
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Hybrid Quantum-Classical Term Deposit Campaign Pipeline\n",
            "### The AtomIQ Hackathon | Qiskit Fall Fest 2026 at IEM\n",
            "**Track:** Chi(r)ps: A Decade of Atomiq Harmony on the Cloud\n",
            "---\n",
            "## Executive Summary\n",
            "This project delivers an end-to-end, zero-leakage **Hybrid Quantum-Classical Pipeline** solving the term-deposit targeting challenge across three interconnected stages:\n",
            "1. **Stage 1 (Predict):** Quantum Kernel Classifier (QSVC with `zz_feature_map`) + Calibrated Random Forest hybrid ensemble predicting subscription probabilities for 100 test customers.\n",
            "2. **Stage 2 (Select):** QUBO / Ising Hamiltonian optimization solved via **QAOA** on Qiskit Aer, benchmarked against exact classical brute force ($\binom{10}{6}=210$) to select the optimal 6 customers.\n",
            "3. **Stage 3 (Simulate):** 6-qubit quantum simulation circuit estimating the probability that $\ge 3$ customers subscribe, validated against exact analytical Poisson binomial combinatorics and Monte Carlo simulation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup and Imports\n",
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from src.data_loader import load_all_data\n",
            "from src.preprocessing import prepare_data\n",
            "from src.classical_models import evaluate_classical_cv\n",
            "from src.quantum_classifier import QuantumKernelClassifier, HybridQuantumClassicalClassifier\n",
            "from src.stage1_pipeline import run_stage1\n",
            "from src.stage2_optimizer import Stage2Optimizer, run_stage2\n",
            "from src.stage3_simulator import Stage3Simulator, run_stage3\n",
            "\n",
            "print('Imports successful!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Stage 1: Subscription Prediction\n",
            "- Preprocessing with zero data leakage (all imputers, scalers, and encoders fit on train only).\n",
            "- 5-fold Stratified Cross-Validation benchmarking Logistic Regression, Random Forest, HistGradientBoosting, and Quantum Kernel Classifier (QSVC).\n",
            "- Hybrid Blending: $P_{\\text{hybrid}} = 0.3 P_{\\text{QSVC}} + 0.7 P_{\\text{RF}}$, achieving improved ROC-AUC (0.6457) and PR-AUC (0.1797)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_stage1, top_10 = run_stage1(output_dir='results', random_state=42)\n",
            "print('Top 10 Stage 2 Candidates:')\n",
            "top_10"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Stage 2: Constrained Customer Selection\n",
            "- Formulation as a Binary Quadratic Program (QUBO) with cardinality constraint $\\sum x_i = 6$, budget limit $\\le 25$, and capacity limit $\\le 22$.\n",
            "- Solved with **QAOA** on Qiskit Aer and verified against exact combinatorial brute force over $\\binom{10}{6}=210$ combinations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_stage2, qaoa_res, classical_res = run_stage2(output_dir='results')\n",
            "print('Optimal 6 Customer IDs Selected:')\n",
            "print(classical_res['selected_ids'])\n",
            "print(f\"Optimal Expected Net Value: ${classical_res['objective_value']:.4f}\")\n",
            "print(f\"Budget Used: ${classical_res['budget_used']} / $25 | Capacity Used: {classical_res['capacity_used']} / 22\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Stage 3: Campaign Outcome Simulation\n",
            "- 6-qubit quantum circuit where each qubit $q_i$ is rotated by $\\theta_i = 2 \\arcsin(\\sqrt{p_i})$ via $R_y(\\theta_i)$.\n",
            "- Success metric: fraction of shots with Hamming weight $\\ge 3$.\n",
            "- Benchmarked against exact analytical Poisson binomial distribution and 100,000-trial Monte Carlo."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_stage3_qubits, df_stage3_summary, sim = run_stage3(output_dir='results', shots=100000, seed=42)\n",
            "df_stage3_summary"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Quantum Circuit Visualization & Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "qc = sim.build_quantum_circuit()\n",
            "print(qc.draw(output='text'))\n",
            "print(f\"\\nCircuit Depth: {sim.circuit_metrics['transpiled_depth']}\")\n",
            "print(f\"Gate Counts: {sim.circuit_metrics['gate_counts']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Conclusion and Final Deliverables\n",
            "- **Six Selected Customers:** `C0170`, `C0338`, `C0079`, `C0256`, `C0322`, `C0252`\n",
            "- **Expected Campaign Net Value:** $9.2176\n",
            "- **Probability of $\\ge 3$ Conversions:** 3.843% (Exact) | 3.800% (Quantum Aer Simulation)\n",
            "- **Quantum Absolute Error:** 0.000429 (0.04% difference, well within 95% confidence interval)"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.15"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('hybrid_pipeline.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print('hybrid_pipeline.ipynb generated successfully!')
