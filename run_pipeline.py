"""
Master Execution Pipeline for Hybrid Quantum-Classical Term Deposit Campaign.
Runs Stage 1, Stage 2, and Stage 3 sequentially, generating all official submission files.
"""

import os
import sys
import time

from src.stage1_pipeline import run_stage1
from src.stage2_optimizer import run_stage2
from src.stage3_simulator import run_stage3

def main():
    print("*" * 75)
    print("HYBRID QUANTUM-CLASSICAL TERM DEPOSIT CAMPAIGN PIPELINE")
    print("The AtomIQ Hackathon | Qiskit Fall Fest 2026 at IEM")
    print("*" * 75)
    
    start_time = time.time()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute Stage 1
    t0 = time.time()
    df_stage1, top_10 = run_stage1(output_dir=output_dir, random_state=42)
    t1 = time.time()
    print(f"\n>>> Stage 1 completed in {t1 - t0:.2f} seconds.\n")
    
    # Execute Stage 2
    t0 = time.time()
    df_stage2, q_res, c_res = run_stage2(output_dir=output_dir)
    t1 = time.time()
    print(f"\n>>> Stage 2 completed in {t1 - t0:.2f} seconds.\n")
    
    # Execute Stage 3
    t0 = time.time()
    df_stage3_qubits, df_stage3_summary, simulator = run_stage3(output_dir=output_dir, shots=100000, seed=42)
    t1 = time.time()
    print(f"\n>>> Stage 3 completed in {t1 - t0:.2f} seconds.\n")
    
    total_time = time.time() - start_time
    print("*" * 75)
    print("FINAL SUMMARY & DELIVERABLES")
    print("*" * 75)
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds.")
    print("\n1. SIX CUSTOMERS SELECTED TO CONTACT:")
    print(f"   {c_res['selected_ids']}")
    print(f"   Expected Net Value:  ${c_res['objective_value']:.4f}")
    print(f"   Total Contact Cost:  ${c_res['budget_used']} (Budget Limit: $25)")
    print(f"   Total Capacity Used: {c_res['capacity_used']} units (Capacity Limit: 22 units)")
    
    print("\n2. PROBABILITY THAT AT LEAST THREE SUBSCRIBE (P >= 3):")
    print(f"   Exact Analytical Result: {df_stage3_summary.loc[0, 'exact_analytical_result']:.6f} ({df_stage3_summary.loc[0, 'exact_analytical_result'] * 100:.3f}%)")
    print(f"   Quantum Estimate (Ideal): {df_stage3_summary.loc[0, 'quantum_estimate_ideal']:.6f} ({df_stage3_summary.loc[0, 'quantum_estimate_ideal'] * 100:.3f}%)")
    print(f"   Quantum Estimate (Noisy): {df_stage3_summary.loc[0, 'quantum_estimate_noisy']:.6f} ({df_stage3_summary.loc[0, 'quantum_estimate_noisy'] * 100:.3f}%)")
    print(f"   Classical Monte Carlo:   {df_stage3_summary.loc[0, 'monte_carlo_result']:.6f} ({df_stage3_summary.loc[0, 'monte_carlo_result'] * 100:.3f}%)")
    print(f"   Quantum Absolute Error:  {df_stage3_summary.loc[0, 'quantum_absolute_error']:.6f}")
    
    print("\n3. OFFICIAL SUBMISSION FILES GENERATED IN 'results/':")
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        print(f"   - {f:<32} ({os.path.getsize(path):,} bytes)")
    print("*" * 75)

if __name__ == '__main__':
    main()
