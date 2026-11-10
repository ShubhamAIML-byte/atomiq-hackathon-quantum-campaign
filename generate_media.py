"""
Generates high-resolution 3:2 ratio project media images for Devpost Image Gallery.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('media', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------
# IMAGE 1: End-to-End System Architecture (3:2 Ratio)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(6, 7.6, "AtomiqDeposit: Hybrid Quantum-Classical Pipeline Architecture", 
        ha='center', va='center', fontsize=18, fontweight='bold', color='#1a237e')
ax.text(6, 7.2, "The AtomIQ Hackathon | Qiskit Fall Fest 2026 at IEM", 
        ha='center', va='center', fontsize=12, color='#455a64', style='italic')

# Stage 1 Card
card1 = patches.FancyBboxPatch((0.5, 2.0), 3.4, 4.7, boxstyle="round,pad=0.2", 
                               ec="#1e88e5", fc="#e3f2fd", lw=2)
ax.add_patch(card1)
ax.text(2.2, 6.3, "STAGE 1: PREDICT", ha='center', va='center', fontsize=13, fontweight='bold', color='#0d47a1')
ax.text(2.2, 5.9, "Subscription Probability", ha='center', va='center', fontsize=10, color='#1565c0')
stage1_text = (
    "• 300 Train / 100 Test Leads\n"
    "• Zero Data Leakage Preprocessing\n"
    "• PCA (d=6) scaled to [0, π]\n"
    "• Qiskit zz_feature_map\n"
    "• Quantum Kernel (QSVC)\n"
    "• Hybrid Blend (0.3 Q + 0.7 RF)\n"
    "• ROC-AUC: 0.6457 | PR-AUC: 0.1797\n\n"
    "OUTPUT: Top 10 Candidates"
)
ax.text(0.7, 4.3, stage1_text, ha='left', va='center', fontsize=9.5, color='#263238', linespacing=1.4)

# Stage 2 Card
card2 = patches.FancyBboxPatch((4.3, 2.0), 3.4, 4.7, boxstyle="round,pad=0.2", 
                               ec="#43a047", fc="#e8f5e9", lw=2)
ax.add_patch(card2)
ax.text(6.0, 6.3, "STAGE 2: SELECT", ha='center', va='center', fontsize=13, fontweight='bold', color='#1b5e20')
ax.text(6.0, 5.9, "Constrained Optimization", ha='center', va='center', fontsize=10, color='#2e7d32')
stage2_text = (
    "• Input: Top 10 Candidates\n"
    "• Maximize Net Value: p_i*V_i - C_i\n"
    "• Exact Cardinality: Σ x_i = 6\n"
    "• Budget: Σ C_i x_i ≤ $25\n"
    "• Capacity: Σ H_i x_i ≤ 22\n"
    "• QUBO / 10-Qubit Ising Model\n"
    "• QAOA on Qiskit Aer (p=1)\n"
    "• Brute-Force C(10, 6) = 210\n\n"
    "OUTPUT: Exact 6 Customers"
)
ax.text(4.5, 4.3, stage2_text, ha='left', va='center', fontsize=9.5, color='#263238', linespacing=1.4)

# Stage 3 Card
card3 = patches.FancyBboxPatch((8.1, 2.0), 3.4, 4.7, boxstyle="round,pad=0.2", 
                               ec="#8e24aa", fc="#f3e5f5", lw=2)
ax.add_patch(card3)
ax.text(9.8, 6.3, "STAGE 3: SIMULATE", ha='center', va='center', fontsize=13, fontweight='bold', color='#4a148c')
ax.text(9.8, 5.9, "Campaign Outcome Risk", ha='center', va='center', fontsize=10, color='#6a1b9a')
stage3_text = (
    "• Input: The 6 Selected Leads\n"
    "• 6-Qubit Quantum Circuit\n"
    "• State Prep: Ry(θ_i)|0>\n"
    "• θ_i = 2 arcsin(sqrt(p_i))\n"
    "• 100,000 Shots (Qiskit Aer)\n"
    "• Success: Hamming Weight ≥ 3\n"
    "• Poisson Binomial Benchmark\n"
    "• Monte Carlo (100k Trials)\n\n"
    "OUTPUT: P(X ≥ 3) = 3.84%"
)
ax.text(8.3, 4.3, stage3_text, ha='left', va='center', fontsize=9.5, color='#263238', linespacing=1.4)

# Arrows between cards
ax.annotate('', xy=(4.2, 4.35), xytext=(3.9, 4.35),
            arrowprops=dict(facecolor='#0288d1', edgecolor='#0288d1', arrowstyle="-|>", lw=3, mutation_scale=20))
ax.annotate('', xy=(8.0, 4.35), xytext=(7.7, 4.35),
            arrowprops=dict(facecolor='#388e3c', edgecolor='#388e3c', arrowstyle="-|>", lw=3, mutation_scale=20))

# Bottom Results Banner
banner = patches.FancyBboxPatch((0.5, 0.4), 11.0, 1.2, boxstyle="round,pad=0.2", 
                                ec="#ff6f00", fc="#fff8e1", lw=2)
ax.add_patch(banner)
ax.text(6.0, 1.15, "FINAL DELIVERABLE ANSWERS", ha='center', va='center', fontsize=12, fontweight='bold', color='#e65100')
summary_line = (
    "Selected Customers: [C0170, C0338, C0079, C0256, C0322, C0252]  |  Expected Net Value: $9.22\n"
    "Budget Consumed: $10 / $25  |  Capacity: 13 / 22 units  |  Conversion Target P(X ≥ 3): 3.843% (Exact) / 3.800% (Quantum Aer)"
)
ax.text(6.0, 0.75, summary_line, ha='center', va='center', fontsize=10, color='#bf360c', linespacing=1.3)

plt.tight_layout()
plt.savefig('media/01_pipeline_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated media/01_pipeline_architecture.png")


# -------------------------------------------------------------
# IMAGE 2: Stage 1 Model Performance Comparison (3:2 Ratio)
# -------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
fig.suptitle("Stage 1: Subscription Prediction — Classical vs. Quantum vs. Hybrid Performance", 
             fontsize=15, fontweight='bold', color='#1a237e', y=0.98)

models = ['Logistic\nRegression', 'Random\nForest', 'HistGradient\nBoosting', 'Quantum Kernel\n(QSVC)', 'Hybrid Ensemble\n(0.3 Q + 0.7 RF)']
colors = ['#90caf9', '#64b5f6', '#42a5f5', '#ab47bc', '#ff9800']

# 1. ROC-AUC
roc_vals = [0.5639, 0.6400, 0.5293, 0.5730, 0.6457]
bars1 = ax1.bar(models, roc_vals, color=colors, edgecolor='black', linewidth=0.8)
ax1.set_ylim(0.45, 0.70)
ax1.set_ylabel("ROC-AUC Score", fontweight='bold')
ax1.set_title("Discriminative Power (ROC-AUC)", fontweight='bold', color='#0d47a1')
for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. PR-AUC
pr_vals = [0.1417, 0.1760, 0.1496, 0.1688, 0.1797]
bars2 = ax2.bar(models, pr_vals, color=colors, edgecolor='black', linewidth=0.8)
ax2.set_ylim(0.10, 0.21)
ax2.set_ylabel("PR-AUC (Avg Precision)", fontweight='bold')
ax2.set_title("Precision-Recall AUC (High Class Imbalance)", fontweight='bold', color='#0d47a1')
for b in bars2:
    h = b.get_height()
    ax2.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Precision@10 & Recall@10
x = np.arange(len(models))
w = 0.35
p10 = [0.20, 0.00, 0.30, 0.10, 0.10]
r10 = [0.0571, 0.00, 0.0857, 0.0286, 0.0286]
ax3.bar(x - w/2, p10, w, label='Precision@10', color='#26a69a', edgecolor='black', lw=0.8)
ax3.bar(x + w/2, r10, w, label='Recall@10', color='#ffa726', edgecolor='black', lw=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_ylabel("Metric Value", fontweight='bold')
ax3.set_title("Top-10 Candidate Ranking Metrics", fontweight='bold', color='#0d47a1')
ax3.legend(loc='upper right')

# 4. Calibration & Probability Quality (Brier Score, lower is better)
brier_vals = [0.1028, 0.1018, 0.1030, 0.1025, 0.1019]
bars4 = ax4.bar(models, brier_vals, color=colors, edgecolor='black', linewidth=0.8)
ax4.set_ylim(0.095, 0.106)
ax4.set_ylabel("Brier Score (Lower is Better)", fontweight='bold')
ax4.set_title("Probability Calibration Quality (Brier Score)", fontweight='bold', color='#0d47a1')
for b in bars4:
    h = b.get_height()
    ax4.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('media/02_stage1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated media/02_stage1_model_comparison.png")


# -------------------------------------------------------------
# IMAGE 3: Stage 2 Optimization & Portfolio Selection (3:2 Ratio)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
fig.suptitle("Stage 2: Constrained Customer Selection via QAOA & Exact Brute Force", 
             fontsize=15, fontweight='bold', color='#1a237e', y=0.98)

# Left: Expected Net Values of Top 10 Candidates
top10_ids = ['C0170', 'C0338', 'C0193', 'C0081', 'C0079', 'C0183', 'C0256', 'C0188', 'C0322', 'C0252']
net_vals = [2.3399, 0.5230, -0.7576, -1.6394, 0.3146, -0.0946, 1.9864, -0.2502, 2.8623, 1.1913]
selected_mask = [True, True, False, False, True, False, True, False, True, True]
bar_colors = ['#2e7d32' if s else '#c62828' for s in selected_mask]

bars = ax1.barh(top10_ids[::-1], net_vals[::-1], color=bar_colors[::-1], edgecolor='black', lw=0.8)
ax1.axvline(0, color='black', lw=1.2, ls='--')
ax1.set_xlabel("Expected Net Value ($)", fontweight='bold')
ax1.set_title("Top 10 Candidates Net Value (Green = Selected 6)", fontweight='bold', color='#1b5e20')
for b in bars:
    w = b.get_width()
    offset = 0.1 if w >= 0 else -0.35
    ax1.annotate(f"${w:.2f}", xy=(w, b.get_y() + b.get_height() / 2),
                 xytext=(offset, 0), textcoords="offset points", ha='left' if w >= 0 else 'right', 
                 va='center', fontsize=9, fontweight='bold')

# Right: Budget & Capacity Resource Allocation
resources = ['Contact Budget ($)', 'Branch Capacity (Units)']
used = [10, 13]
limits = [25, 22]
y_pos = np.arange(len(resources))

ax2.barh(y_pos, limits, color='#e0e0e0', edgecolor='black', lw=1, height=0.45, label='Constraint Limit')
ax2.barh(y_pos, used, color='#1976d2', edgecolor='black', lw=1, height=0.45, label='Consumed by Selected 6')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(resources, fontweight='bold')
ax2.set_xlabel("Resource Amount", fontweight='bold')
ax2.set_title("Resource Consumption vs Constraint Limits", fontweight='bold', color='#0d47a1')
ax2.legend(loc='lower right')

ax2.annotate("Budget: $10 / $25 (60% Free)", xy=(10, 0), xytext=(12, 0),
             ha='left', va='center', fontweight='bold', color='#0d47a1', fontsize=10)
ax2.annotate("Capacity: 13 / 22 (41% Free)", xy=(13, 1), xytext=(14.5, 1),
             ha='left', va='center', fontweight='bold', color='#0d47a1', fontsize=10)

# Summary Box inside right plot
summary_text = (
    "OPTIMIZATION SUMMARY:\n"
    "• Combinatorial Space: C(10, 6) = 210 combinations\n"
    "• Feasible Space: 210 / 210 feasible\n"
    "• Total Portfolio Value: $9.2176\n"
    "• Optimal Bitstring: 1100101011\n"
    "• QAOA Parameters: γ* = 1.3631, β* = 0.7847\n"
    "• Feasibility Status: 100% Valid & Certified"
)
ax2.text(1, -0.6, summary_text, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#43a047'))

plt.tight_layout()
plt.savefig('media/03_stage2_qaoa_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated media/03_stage2_qaoa_optimization.png")


# -------------------------------------------------------------
# IMAGE 4: Stage 3 Quantum Simulation & Benchmark Distribution (3:2 Ratio)
# -------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
fig.suptitle("Stage 3: 6-Qubit Quantum Campaign Simulation & Conversion Benchmark", 
             fontsize=15, fontweight='bold', color='#4a148c', y=0.98)

# 1. Quantum State Preparation Angles
qubits = [f"q[{i}]\n{cid}" for i, cid in enumerate(['C0170', 'C0338', 'C0079', 'C0256', 'C0322', 'C0252'])]
thetas = [0.8005, 0.7675, 0.7615, 0.7547, 0.7471, 0.7465]
probs = [0.1518, 0.1402, 0.1381, 0.1357, 0.1332, 0.1330]

bars = ax1.bar(qubits, thetas, color='#ba68c8', edgecolor='black', lw=0.8)
ax1.set_ylabel("Rotation Angle θ (rad)", fontweight='bold')
ax1.set_title("Single-Qubit State Prep: Ry(θ_i)|0>", fontweight='bold', color='#4a148c')
ax1.set_ylim(0, 1.0)
for i, b in enumerate(bars):
    ax1.annotate(f"p={probs[i]:.3f}\nθ={thetas[i]:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Probability Mass Function P(X = k)
k_vals = np.arange(7)
dist_exact = [0.408282, 0.394531, 0.158758, 0.034052, 0.004106, 0.000264, 0.000007]
ax2.plot(k_vals, dist_exact, 'o-', color='#7b1fa2', lw=2, markersize=7, label='Poisson Binomial PMF')
ax2.fill_between(k_vals[3:], dist_exact[3:], color='#e1bee7', alpha=0.6, label='Success Zone (X ≥ 3)')
ax2.set_xlabel("Number of Subscribing Customers (k)", fontweight='bold')
ax2.set_ylabel("Probability P(X = k)", fontweight='bold')
ax2.set_title("Exact Probability Distribution of Conversions", fontweight='bold', color='#4a148c')
ax2.set_xticks(k_vals)
ax2.legend()
for k, p in enumerate(dist_exact):
    if p > 0.001:
        ax2.annotate(f"{p*100:.1f}%", xy=(k, p), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)

# 3. Method Comparison for P(X >= 3)
methods = ['Exact Analytical\n(Poisson Binomial)', 'Quantum Aer\n(100k Shots)', 'Monte Carlo\n(100k Trials)', 'Noisy Quantum\n(Depol + Readout)']
est_vals = [0.038429, 0.038000, 0.038760, 0.043510]
colors_m = ['#6a1b9a', '#8e24aa', '#26a69a', '#ef5350']
bars3 = ax3.bar(methods, [v * 100 for v in est_vals], color=colors_m, edgecolor='black', lw=0.8)
ax3.set_ylabel("Probability P(X ≥ 3) [%]", fontweight='bold')
ax3.set_title("Benchmarking P(X ≥ 3) Across Methods", fontweight='bold', color='#4a148c')
ax3.set_ylim(0, 5.5)
for b in bars3:
    h = b.get_height()
    ax3.annotate(f"{h:.3f}%", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Error Metrics & Confidence Interval
err_labels = ['Quantum Error\n(vs. Exact)', 'Monte Carlo Error\n(vs. Exact)', 'Quantum 95% CI\nHalf-Width']
err_vals = [0.000429 * 100, 0.000331 * 100, 0.001185 * 100]
bars4 = ax4.bar(err_labels, err_vals, color=['#ab47bc', '#4db6ac', '#ffd54f'], edgecolor='black', lw=0.8)
ax4.set_ylabel("Absolute Margin [%]", fontweight='bold')
ax4.set_title("Precision & Statistical Error Margins", fontweight='bold', color='#4a148c')
for b in bars4:
    h = b.get_height()
    ax4.annotate(f"{h:.4f}%", xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('media/04_stage3_quantum_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated media/04_stage3_quantum_simulation.png")
