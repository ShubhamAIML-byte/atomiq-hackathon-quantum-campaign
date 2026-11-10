"""
Stage 3: Campaign Outcome Simulation.
Simulates campaign conversion probability using a 6-qubit quantum circuit on Qiskit Aer,
benchmarks against exact analytical Poisson binomial distribution and Monte Carlo,
performs comprehensive error analysis, and outputs the Stage 3 result file.
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

class Stage3Simulator:
    """
    Quantum Circuit Simulation and Analytical Benchmarking of Campaign Outcomes.
    """
    def __init__(self, selected_df, shots=100000, seed=42):
        self.selected_df = selected_df.copy().reset_index(drop=True)
        self.shots = shots
        self.seed = seed
        
        self.selected_ids = self.selected_df['customer_id'].tolist()
        self.probs = self.selected_df['predicted_probability'].values
        self.num_qubits = len(self.selected_ids)
        self.thetas = [float(2.0 * np.arcsin(np.sqrt(p))) for p in self.probs]
        
        self.exact_prob = None
        self.quantum_prob = None
        self.noisy_quantum_prob = None
        self.mc_prob = None
        self.circuit_metrics = None

    def compute_exact_poisson_binomial(self):
        """
        Calculates exact analytical P(X >= 3) via probability generating polynomial convolution.
        """
        poly = np.array([1.0])
        for p in self.probs:
            poly = np.convolve(poly, [1.0 - p, p])
            
        # poly[k] corresponds to P(X = k)
        prob_ge_3 = float(np.sum(poly[3:]))
        self.exact_prob = prob_ge_3
        self.exact_dist = {f"P(X={k})": float(poly[k]) for k in range(len(poly))}
        return self.exact_prob, self.exact_dist

    def run_monte_carlo(self, trials=100000):
        """
        Runs classical Monte Carlo simulation over independent Bernoulli trials.
        """
        np.random.seed(self.seed)
        samples = np.random.binomial(1, self.probs, size=(trials, self.num_qubits))
        successes = np.sum(np.sum(samples, axis=1) >= 3)
        self.mc_prob = float(successes / trials)
        return self.mc_prob

    def build_quantum_circuit(self):
        """
        Constructs the 6-qubit quantum simulation circuit.
        Qubit i is prepared into: cos(theta_i/2)|0> + sin(theta_i/2)|1>
        such that P(|1>) = sin^2(theta_i/2) = p_i.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits, name="CampaignSimulation")
        for i in range(self.num_qubits):
            qc.ry(self.thetas[i], i)
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run_quantum_simulation(self):
        """
        Executes ideal and noisy finite-shot simulation on Qiskit Aer.
        """
        qc = self.build_quantum_circuit()
        
        # Transpiled circuit metrics
        sim_ideal = AerSimulator()
        transpiled_qc = transpile(qc, sim_ideal, optimization_level=1)
        depth = transpiled_qc.depth()
        gate_counts = dict(transpiled_qc.count_ops())
        
        # 1. Ideal simulation
        result_ideal = sim_ideal.run(
            transpiled_qc,
            shots=self.shots,
            seed_simulator=self.seed
        ).result()
        counts_ideal = result_ideal.get_counts()
        
        # Count successes: bitstrings with Hamming weight >= 3
        # In Qiskit, bitstrings are ordered right-to-left: q5 q4 q3 q2 q1 q0
        success_shots_ideal = sum(
            count for bitstr, count in counts_ideal.items() if bitstr.count('1') >= 3
        )
        self.quantum_prob = float(success_shots_ideal / self.shots)
        self.counts_ideal = counts_ideal
        
        # 2. Noisy simulation (realistic depolarizing + readout noise)
        noise_model = NoiseModel()
        dep_err = depolarizing_error(0.001, 1)
        noise_model.add_all_qubit_quantum_error(dep_err, ['ry'])
        ro_err = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
        noise_model.add_all_qubit_readout_error(ro_err)
        
        sim_noisy = AerSimulator(noise_model=noise_model)
        result_noisy = sim_noisy.run(
            transpiled_qc,
            shots=self.shots,
            seed_simulator=self.seed
        ).result()
        counts_noisy = result_noisy.get_counts()
        success_shots_noisy = sum(
            count for bitstr, count in counts_noisy.items() if bitstr.count('1') >= 3
        )
        self.noisy_quantum_prob = float(success_shots_noisy / self.shots)
        
        self.circuit_metrics = {
            'logical_qubits': self.num_qubits,
            'physical_qubits': self.num_qubits,
            'original_depth': qc.depth(),
            'transpiled_depth': depth,
            'gate_counts': gate_counts,
            'swap_count': gate_counts.get('swap', 0),
            'two_qubit_gates': 0,
            'shots': self.shots
        }
        
        return self.quantum_prob, self.noisy_quantum_prob

def run_stage3(output_dir='results', shots=100000, seed=42):
    """
    Executes Stage 3 end-to-end.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STAGE 3: CAMPAIGN OUTCOME SIMULATION")
    print("=" * 60)
    
    # Load Stage 2 results to retrieve selected 6 customers
    stage2_path = os.path.join(output_dir, 'stage2_result.csv')
    df_stage2 = pd.read_csv(stage2_path)
    
    # We take the 6 selected customers (from optimal selection)
    selected_6 = df_stage2[df_stage2['selected_by_classical'] == 1].copy().reset_index(drop=True)
    print("Selected 6 Customers for Stage 3 Simulation:")
    print("-" * 65)
    print(f"{'Qubit':<8}{'Customer ID':<15}{'Stage 1 Prob':<18}{'Rotation Theta (rad)':<22}")
    print("-" * 65)
    thetas = [2.0 * np.arcsin(np.sqrt(p)) for p in selected_6['predicted_probability']]
    for i, row in selected_6.iterrows():
        print(f"q[{i}]    {row['customer_id']:<15}{row['predicted_probability']:<18.4f}{thetas[i]:<22.4f}")
    print("-" * 65)
    
    simulator = Stage3Simulator(selected_6, shots=shots, seed=seed)
    
    # 1. Exact Analytical Calculation
    p_exact, dist = simulator.compute_exact_poisson_binomial()
    print(f"\n[Stage 3.1] Exact Analytical Calculation (Poisson Binomial):")
    print(f"  P(Subscribers >= 3) = {p_exact:.6f} ({p_exact * 100:.3f}%)")
    
    # 2. Quantum Circuit Simulation
    p_quantum, p_noisy = simulator.run_quantum_simulation()
    print(f"\n[Stage 3.2] Quantum Circuit Simulation (Qiskit Aer, shots = {shots:,}):")
    print(f"  Ideal Quantum Estimate: {p_quantum:.6f} ({p_quantum * 100:.3f}%)")
    print(f"  Noisy Quantum Estimate: {p_noisy:.6f} ({p_noisy * 100:.3f}%)")
    print(f"  Circuit Depth: {simulator.circuit_metrics['transpiled_depth']}, Gate Counts: {simulator.circuit_metrics['gate_counts']}")
    
    # 3. Classical Monte Carlo Simulation
    p_mc = simulator.run_monte_carlo(trials=shots)
    print(f"\n[Stage 3.3] Classical Monte Carlo Simulation (trials = {shots:,}):")
    print(f"  Monte Carlo Estimate:   {p_mc:.6f} ({p_mc * 100:.3f}%)")
    
    # 4. Error Analysis
    quantum_abs_err = abs(p_quantum - p_exact)
    quantum_rel_err = (quantum_abs_err / p_exact) * 100
    mc_abs_err = abs(p_mc - p_exact)
    std_err = np.sqrt(p_quantum * (1 - p_quantum) / shots)
    ci_95 = 1.96 * std_err
    
    print("\n[Stage 3.4] Comprehensive Error Analysis:")
    print(f"  Quantum Absolute Error:     {quantum_abs_err:.6f}")
    print(f"  Quantum Relative Error:     {quantum_rel_err:.2f}%")
    print(f"  Monte Carlo Absolute Error: {mc_abs_err:.6f}")
    print(f"  Quantum 95% Confidence Int: [{p_quantum - ci_95:.6f}, {p_quantum + ci_95:.6f}]")
    
    # 5. Build Stage 3 Result File
    # Format per submission guidelines:
    # "selected IDs, qubit order, the six probabilities, circuit parameters,
    #  measurement counts, your quantum estimate, the exact result, the Monte Carlo result, and the errors."
    result_rows = []
    for i, row in selected_6.iterrows():
        result_rows.append({
            'qubit_index': i,
            'customer_id': row['customer_id'],
            'stage1_probability': float(row['predicted_probability']),
            'circuit_rotation_theta_rad': float(thetas[i]),
            'circuit_gate': 'ry',
            'bit_ordering_convention': f"q[{i}] is bit {i} from right (little-endian)"
        })
    df_stage3_qubits = pd.DataFrame(result_rows)
    
    summary_stage3 = pd.DataFrame([{
        'quantum_estimate_ideal': p_quantum,
        'quantum_estimate_noisy': p_noisy,
        'exact_analytical_result': p_exact,
        'monte_carlo_result': p_mc,
        'quantum_absolute_error': quantum_abs_err,
        'quantum_relative_error_pct': quantum_rel_err,
        'monte_carlo_absolute_error': mc_abs_err,
        'simulation_shots': shots,
        'circuit_transpiled_depth': simulator.circuit_metrics['transpiled_depth'],
        'two_qubit_gate_count': simulator.circuit_metrics['two_qubit_gates'],
        'swap_count': simulator.circuit_metrics['swap_count']
    }])
    
    output_path = os.path.join(output_dir, 'stage3_result.csv')
    summary_path = os.path.join(output_dir, 'stage3_summary.csv')
    counts_path = os.path.join(output_dir, 'stage3_measurement_counts.json')
    
    df_stage3_qubits.to_csv(output_path, index=False)
    summary_stage3.to_csv(summary_path, index=False)
    with open(counts_path, 'w') as f:
        json.dump(simulator.counts_ideal, f, indent=2)
        
    print(f"\nStage 3 result saved to: {output_path}")
    print(f"Stage 3 summary saved to: {summary_path}")
    print(f"Measurement counts saved to: {counts_path}")
    
    return df_stage3_qubits, summary_stage3, simulator

if __name__ == '__main__':
    df_qubits, df_summary, sim = run_stage3()
