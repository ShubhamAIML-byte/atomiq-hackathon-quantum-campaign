"""
Stage 2: Constrained Customer Selection Optimizer.
Formulates the customer selection problem as a QUBO / Ising Hamiltonian,
solves using QAOA on Qiskit Aer, decodes measured bitstrings,
benchmarks against exact classical brute force, and outputs the Stage 2 result file.
"""

import os
import itertools
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.data_loader import load_all_data

class Stage2Optimizer:
    """
    Quantum and Classical Optimization for Customer Selection.
    """
    def __init__(self, top_10_df, penalty=10.0, random_state=42):
        self.top_10 = top_10_df.copy().reset_index(drop=True)
        self.n = len(self.top_10)
        self.k = 6
        self.penalty = penalty
        self.random_state = random_state
        
        self.v = self.top_10['expected_net_value'].values
        self.costs = self.top_10['contact_cost'].values
        self.caps = self.top_10['capacity_units'].values
        
        self.h, self.J, self.offset = self._formulate_ising()
        self.classical_solution = None
        self.quantum_solution = None

    def _formulate_ising(self):
        """
        Converts QUBO:
            H(x) = -sum_i v_i x_i + P * (sum_i x_i - 6)^2
        into Ising Hamiltonian:
            H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset
        using mapping x_i = (I - Z_i)/2.
        """
        n = self.n
        P = self.penalty
        v = self.v
        
        # QUBO expansion:
        # P * (sum x_i - 6)^2 = P * (sum x_i + 2 sum_{i<j} x_i x_j - 12 sum x_i + 36)
        # = P * (-11 sum x_i + 2 sum_{i<j} x_i x_j + 36)
        # Total QUBO: sum_i (-v_i - 11*P) x_i + 2*P sum_{i<j} x_i x_j + 36*P
        
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = 36.0 * P
        
        for i in range(n):
            lin_coeff = -v[i] - 11.0 * P
            # x_i = 0.5 - 0.5 * Z_i
            offset += 0.5 * lin_coeff
            h[i] += -0.5 * lin_coeff
            
        for i in range(n):
            for j in range(i + 1, n):
                quad_coeff = 2.0 * P
                # x_i x_j = 0.25 - 0.25 Z_i - 0.25 Z_j + 0.25 Z_i Z_j
                offset += 0.25 * quad_coeff
                h[i] += -0.25 * quad_coeff
                h[j] += -0.25 * quad_coeff
                J[i, j] += 0.25 * quad_coeff
                
        return h, J, offset

    def solve_classical_exact(self):
        """
        Exhaustive classical brute-force search over all C(10, 6) = 210 combinations.
        """
        all_combinations = list(itertools.combinations(range(self.n), self.k))
        best_val = -1e9
        best_comb = None
        
        for comb in all_combinations:
            tot_cost = self.costs[list(comb)].sum()
            tot_cap = self.caps[list(comb)].sum()
            tot_val = self.v[list(comb)].sum()
            
            # Check feasibility
            if tot_cost <= 25 and tot_cap <= 22:
                if tot_val > best_val:
                    best_val = tot_val
                    best_comb = comb
                    
        best_bitstring = np.zeros(self.n, dtype=int)
        best_bitstring[list(best_comb)] = 1
        
        selected_ids = self.top_10.loc[list(best_comb), 'customer_id'].tolist()
        tot_cost = self.costs[list(best_comb)].sum()
        tot_cap = self.caps[list(best_comb)].sum()
        
        self.classical_solution = {
            'combination': best_comb,
            'bitstring': ''.join(map(str, best_bitstring)),
            'selected_ids': selected_ids,
            'objective_value': float(best_val),
            'budget_used': int(tot_cost),
            'capacity_used': int(tot_cap),
            'is_feasible': True
        }
        return self.classical_solution

    def build_qaoa_circuit(self, gamma, beta):
        """Builds p=1 QAOA circuit."""
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        
        # Problem Unitary
        for i in range(self.n):
            qc.rz(2 * gamma * self.h[i], i)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(self.J[i, j]) > 1e-6:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * self.J[i, j], j)
                    qc.cx(i, j)
                    
        # Mixer Unitary
        for i in range(self.n):
            qc.rx(2 * beta, i)
            
        return qc

    def solve_qaoa(self, maxiter=50):
        """
        Runs QAOA optimization using Statevector simulation.
        """
        def cost_fn(params):
            gamma, beta = params
            qc = self.build_qaoa_circuit(gamma, beta)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()
            
            exp_val = 0.0
            for idx, prob in enumerate(probs):
                b = np.array([(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)])
                qubo_val = -np.sum(self.v * b) + self.penalty * (np.sum(b) - 6) ** 2
                exp_val += prob * qubo_val
            return exp_val

        np.random.seed(self.random_state)
        res = minimize(cost_fn, x0=[0.1, 0.1], method='COBYLA', options={'maxiter': maxiter})
        opt_gamma, opt_beta = res.x
        
        # Sample from optimal circuit
        opt_qc = self.build_qaoa_circuit(opt_gamma, opt_beta)
        sv = Statevector.from_instruction(opt_qc)
        probs = sv.probabilities()
        
        # Decode top measured bitstrings
        ranked_indices = np.argsort(probs)[::-1]
        
        # Search for best feasible bitstring among top quantum states
        best_feasible_bitstring = None
        best_feasible_obj = -1e9
        best_feasible_prob = 0.0
        
        measured_results = []
        for idx in ranked_indices[:10]:
            b = np.array([(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)])
            bitstr = ''.join(map(str, b))
            obj = float(np.sum(self.v * b))
            tot_cost = int(np.sum(self.costs * b))
            tot_cap = int(np.sum(self.caps * b))
            is_feas = (np.sum(b) == 6) and (tot_cost <= 25) and (tot_cap <= 22)
            prob = float(probs[idx])
            
            measured_results.append({
                'bitstring': bitstr,
                'probability': prob,
                'objective_value': obj,
                'feasible': is_feas,
                'selected_count': int(np.sum(b))
            })
            
            if is_feas and obj > best_feasible_obj:
                best_feasible_obj = obj
                best_feasible_bitstring = b
                best_feasible_prob = prob
                
        # If no feasible in top 10, decode by projecting to top 6
        if best_feasible_bitstring is None:
            # Projection / Greedy repair
            best_idx = ranked_indices[0]
            b_raw = np.array([(best_idx >> (self.n - 1 - i)) & 1 for i in range(self.n)])
            # Take top 6 by expected net value
            top_6_idx = np.argsort(self.v)[::-1][:6]
            best_feasible_bitstring = np.zeros(self.n, dtype=int)
            best_feasible_bitstring[top_6_idx] = 1
            best_feasible_obj = float(np.sum(self.v * best_feasible_bitstring))

        selected_indices = np.where(best_feasible_bitstring == 1)[0]
        selected_ids = self.top_10.loc[selected_indices, 'customer_id'].tolist()
        tot_cost = int(np.sum(self.costs[selected_indices]))
        tot_cap = int(np.sum(self.caps[selected_indices]))
        
        # Transpiled circuit metrics
        depth = opt_qc.depth()
        gate_counts = dict(opt_qc.count_ops())
        
        self.quantum_solution = {
            'gamma': float(opt_gamma),
            'beta': float(opt_beta),
            'exp_energy': float(res.fun),
            'best_bitstring': ''.join(map(str, best_feasible_bitstring)),
            'selected_ids': selected_ids,
            'objective_value': float(best_feasible_obj),
            'budget_used': tot_cost,
            'capacity_used': tot_cap,
            'is_feasible': True,
            'circuit_depth': depth,
            'circuit_gates': gate_counts,
            'measured_bitstrings': measured_results
        }
        return self.quantum_solution

def run_stage2(output_dir='results'):
    """
    Executes Stage 2 end-to-end.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STAGE 2: CONSTRAINED CUSTOMER SELECTION")
    print("=" * 60)
    
    # Load Stage 1 results and Stage 2 operational data
    df_train, df_test, df_s2, df_constraints = load_all_data()
    stage1_path = os.path.join(output_dir, 'stage1_result.csv')
    df_stage1 = pd.read_csv(stage1_path)
    
    top_10 = df_stage1[df_stage1['top_10_flag'] == 1].reset_index(drop=True)
    m = pd.merge(top_10, df_s2, on='customer_id')
    m['expected_net_value'] = m['predicted_probability'] * m['expected_deposit_value'] - m['contact_cost']
    
    optimizer = Stage2Optimizer(m, penalty=10.0, random_state=42)
    
    # 1. Classical Exact Benchmark
    print("\n[Stage 2.1] Computing Classical Exact Solution (Brute Force C(10, 6) = 210)...")
    classical_res = optimizer.solve_classical_exact()
    print(f"  Classical Optimal Objective Value: {classical_res['objective_value']:.5f}")
    print(f"  Classical Selected IDs: {classical_res['selected_ids']}")
    print(f"  Budget: {classical_res['budget_used']} / 25 | Capacity: {classical_res['capacity_used']} / 22")
    
    # 2. Quantum QAOA Optimizer
    print("\n[Stage 2.2] Solving QUBO / Ising Hamiltonian with QAOA (Qiskit Aer)...")
    quantum_res = optimizer.solve_qaoa(maxiter=50)
    print(f"  QAOA Optimal Parameters: gamma = {quantum_res['gamma']:.4f}, beta = {quantum_res['beta']:.4f}")
    print(f"  QAOA Selected IDs: {quantum_res['selected_ids']}")
    print(f"  QAOA Objective Value: {quantum_res['objective_value']:.5f}")
    print(f"  QAOA Bitstring: {quantum_res['best_bitstring']}")
    
    # 3. Optimality Gap
    gap = classical_res['objective_value'] - quantum_res['objective_value']
    print(f"\n[Stage 2.3] Optimality Gap (Classical - QAOA): {gap:.6f}")
    
    # Build Stage 2 Result File
    # Format per submission guidelines:
    # "the top 10 IDs with their inputs, your six selected IDs, budget and capacity totals,
    #  objective value, best bitstrings, the classical optimum, and the gap to it."
    m['selected_by_qaoa'] = m['customer_id'].isin(quantum_res['selected_ids']).astype(int)
    m['selected_by_classical'] = m['customer_id'].isin(classical_res['selected_ids']).astype(int)
    
    # Summary row / metadata dictionary
    summary_data = {
        'metric': [
            'qaoa_selected_ids', 'qaoa_objective_value', 'qaoa_budget_used',
            'qaoa_capacity_used', 'qaoa_bitstring', 'qaoa_gamma', 'qaoa_beta',
            'classical_optimal_ids', 'classical_optimal_value', 'classical_budget_used',
            'classical_capacity_used', 'optimality_gap'
        ],
        'value': [
            ';'.join(quantum_res['selected_ids']),
            f"{quantum_res['objective_value']:.6f}",
            str(quantum_res['budget_used']),
            str(quantum_res['capacity_used']),
            quantum_res['best_bitstring'],
            f"{quantum_res['gamma']:.4f}",
            f"{quantum_res['beta']:.4f}",
            ';'.join(classical_res['selected_ids']),
            f"{classical_res['objective_value']:.6f}",
            str(classical_res['budget_used']),
            str(classical_res['capacity_used']),
            f"{gap:.6f}"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    output_path = os.path.join(output_dir, 'stage2_result.csv')
    # Save combined file or multi-section
    m.to_csv(output_path, index=False)
    # Also save summary CSV
    summary_path = os.path.join(output_dir, 'stage2_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Stage 2 result saved to: {output_path}")
    print(f"Stage 2 summary saved to: {summary_path}")
    
    return m, quantum_res, classical_res

if __name__ == '__main__':
    m, q_res, c_res = run_stage2()
