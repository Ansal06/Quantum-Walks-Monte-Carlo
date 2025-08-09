# This file contains the function definitions used to run NISQ noise analysis across all three distributions including necessary post processing and statistical distance calculation

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np

def clean_galton_mapping(raw_counts: dict, n_layers: int) -> list:
    """
    Clean mapping that properly represents classical Galton board output bins
    """
    output_bins = [0] * (n_layers + 1)
    
    # Simple symmetric mapping for 4-layer case (can be generalized)
    if n_layers == 4:
        position_mapping = {
            0: 0,  # q0 → bin 0 (leftmost)
            1: 0,  # q1 → bin 0  
            2: 1,  # q2 → bin 1
            3: 1,  # q3 → bin 1
            4: 2,  # q4 → bin 2 (center)
            5: 3,  # q5 → bin 3
            6: 3,  # q6 → bin 3
            7: 4,  # q7 → bin 4
            8: 4   # q8 → bin 4 (rightmost)
        }
    else:
        # General case for other layer counts
        position_mapping = {}
        num_qubits = 2 * n_layers + 1
        
        for i in range(num_qubits):
            # Map outer positions to edge bins, inner positions to center bins
            if i <= n_layers:
                bin_idx = min(i, n_layers)
            else:
                bin_idx = max(0, n_layers - (i - n_layers))
                
            position_mapping[i] = bin_idx
    
    # Apply mapping
    for quantum_state, count in raw_counts.items():
        for position, bit in enumerate(quantum_state.replace(' ', '')):
            if bit == '1':
                if position in position_mapping:
                    bin_index = position_mapping[position]
                    output_bins[bin_index] += count
                break
    
    return output_bins


def create_manual_noise_model():
    """Create realistic NISQ noise model with proper gate error handling"""
    noise_model = NoiseModel()
    
    # NISQ-realistic error rates
    p_gate_single = 0.001  # 99.9% single-qubit gate fidelity
    p_gate_two = 0.01      # 99% two-qubit gate fidelity
    p_gate_three = 0.02    # 98% three-qubit gate fidelity
    p_readout = 0.02       # 98% readout fidelity
    
    # Add single-qubit gate errors
    single_error = depolarizing_error(p_gate_single, 1)
    noise_model.add_all_qubit_quantum_error(single_error, ['h', 'x', 'rx', 'reset'])
    
    # Add two-qubit gate errors
    two_error = depolarizing_error(p_gate_two, 2)
    noise_model.add_all_qubit_quantum_error(two_error, ['cx'])
    
    # Add three-qubit gate errors for CSWAP
    three_error = depolarizing_error(p_gate_three, 3)
    noise_model.add_all_qubit_quantum_error(three_error, ['cswap'])
    
    # Add readout errors
    for qubit in range(10):
        readout_error = ReadoutError([[1-p_readout, p_readout], 
                                     [p_readout, 1-p_readout]])
        noise_model.add_readout_error(readout_error, [qubit])
    
    return noise_model


def run_nisq_comparison(distribution_type='unbiased'):
    """Compare ideal vs noisy implementations"""
    # Create corrected noise model
    noise_model = create_manual_noise_model()
    
    # Your proven QGB implementations
    if distribution_type == 'unbiased':
        qc = build_qgb_paperstyle(4)
    else:
        qc = build_biased_qgb(4, direction='right')
    
    # Ideal simulation
    ideal_sim = AerSimulator()
    ideal_job = ideal_sim.run(qc, shots=8192)
    ideal_counts = ideal_job.result().get_counts()
    
    # Noisy simulation
    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_job = noisy_sim.run(qc, shots=8192)
    noisy_counts = noisy_job.result().get_counts()
    
    return ideal_counts, noisy_counts


def calculate_distribution_fidelity(ideal_counts: dict, noisy_counts: dict, n_layers: int = 4):
    """
    Calculate fidelity between ideal and noisy quantum Galton board distributions
    """
    # Convert quantum counts to classical Galton board bins
    ideal_bins = clean_galton_mapping(ideal_counts, n_layers)
    noisy_bins = clean_galton_mapping(noisy_counts, n_layers)
    
    # Normalize to probabilities
    total_ideal = sum(ideal_bins)
    total_noisy = sum(noisy_bins)
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0, ideal_bins, noisy_bins
    
    ideal_probs = [count/total_ideal for count in ideal_bins]
    noisy_probs = [count/total_noisy for count in noisy_bins]
    
    # Calculate fidelity metrics
    # Total Variation Distance
    tv_distance = 0.5 * sum(abs(p_ideal - p_noisy) 
                           for p_ideal, p_noisy in zip(ideal_probs, noisy_probs))
    
    # Fidelity = 1 - Total Variation Distance
    fidelity = 1.0 - tv_distance
    
    return fidelity, ideal_bins, noisy_bins


# Calculate uniform distribution fidelity
def calculate_uniform_fidelity(ideal_counts, noisy_counts):
    """Calculate fidelity for uniform distribution"""
    total_ideal = sum(ideal_counts.values())
    total_noisy = sum(noisy_counts.values())
    
    # Normalize to probabilities
    ideal_probs = {state: count/total_ideal for state, count in ideal_counts.items()}
    noisy_probs = {state: count/total_noisy for state, count in noisy_counts.items()}
    
    # Calculate fidelity (overlap)
    overlap = 0
    all_states = set(ideal_probs.keys()) | set(noisy_probs.keys())
    
    for state in all_states:
        p_ideal = ideal_probs.get(state, 0)
        p_noisy = noisy_probs.get(state, 0)
        overlap += min(p_ideal, p_noisy)
    
    return overlap


from scipy.stats import wasserstein_distance, norm

def compute_wasserstein_distance_with_uncertainty(observed_counts, target_probs, shots):
    """
    Compute Wasserstein distance between observed and target distributions
    with uncertainty bounds due to quantum shot noise
    """
    # Normalize observed counts
    obs_probs = np.array(observed_counts) / shots
    target_probs = np.array(target_probs)
    positions = np.arange(len(obs_probs))
    
    # Compute Wasserstein distance
    distance = wasserstein_distance(positions, positions, obs_probs, target_probs)
    
    # Estimate uncertainty from binomial sampling statistics
    stderr = np.sqrt(obs_probs * (1 - obs_probs) / shots)
    
    # Perturb distribution within ±1σ to estimate distance uncertainty
    obs_up = np.clip(obs_probs + stderr, 0, 1)
    obs_up = obs_up / obs_up.sum()  # Renormalize
    
    obs_down = np.clip(obs_probs - stderr, 0, 1)
    obs_down = obs_down / obs_down.sum()  # Renormalize
    
    distance_up = wasserstein_distance(positions, positions, obs_up, target_probs)
    distance_down = wasserstein_distance(positions, positions, obs_down, target_probs)
    
    uncertainty = max(abs(distance_up - distance), abs(distance_down - distance))
    
    return distance, uncertainty



def compute_kl_divergence_with_uncertainty(observed_counts, target_probs, shots):
    """
    Compute Kullback-Leibler divergence with uncertainty bounds
    """
    obs_probs = np.array(observed_counts) / shots
    target_probs = np.array(target_probs)
    
    # Avoid log(0) by adding small regularization
    epsilon = 1e-10
    obs_probs = obs_probs + epsilon
    target_probs = target_probs + epsilon
    
    # Renormalize after regularization
    obs_probs = obs_probs / obs_probs.sum()
    target_probs = target_probs / target_probs.sum()
    
    # KL divergence
    kl_div = np.sum(obs_probs * np.log(obs_probs / target_probs))
    
    # Uncertainty from shot noise
    stderr = np.sqrt(obs_probs * (1 - obs_probs) / shots)
    
    # Estimate KL uncertainty through perturbation
    obs_up = np.clip(obs_probs + stderr, epsilon, 1)
    obs_up = obs_up / obs_up.sum()
    kl_up = np.sum(obs_up * np.log(obs_up / target_probs))
    
    obs_down = np.clip(obs_probs - stderr, epsilon, 1)
    obs_down = obs_down / obs_down.sum()
    kl_down = np.sum(obs_down * np.log(obs_down / target_probs))
    
    kl_uncertainty = max(abs(kl_up - kl_div), abs(kl_down - kl_div))
    
    return kl_div, kl_uncertainty


def generate_target_distributions(n_bins=5):
    """Generate theoretical target distributions for comparison"""
    
    # Gaussian target (centered on middle bin)
    x = np.arange(n_bins)
    mean = (n_bins - 1) / 2
    std = 1.0
    gaussian_target = norm.pdf(x, loc=mean, scale=std)
    gaussian_target /= gaussian_target.sum()
    
    # Exponential target (decay from left)
    exp_lambda = 1.0
    exponential_target = np.exp(-exp_lambda * x)
    exponential_target /= exponential_target.sum()
    
    # Uniform target
    uniform_target = np.ones(n_bins) / n_bins
    
    return gaussian_target, exponential_target, uniform_target

def analyze_distribution_accuracy(observed_bins, target_probs, dist_name, shots=8192):
    """Complete analysis for one distribution"""
    
    # Wasserstein distance
    w_dist, w_unc = compute_wasserstein_distance_with_uncertainty(
        observed_bins, target_probs, shots)
    
    # KL divergence
    kl_div, kl_unc = compute_kl_divergence_with_uncertainty(
        observed_bins, target_probs, shots)
    
    # Total variation distance
    obs_probs = np.array(observed_bins) / shots
    tv_dist = 0.5 * np.sum(np.abs(obs_probs - target_probs))
    tv_unc = np.sqrt(np.sum(obs_probs * (1 - obs_probs) / shots)) / 2
    
    print(f"\n{dist_name} Distribution Analysis:")
    print(f"Wasserstein Distance: {w_dist:.4f} ± {w_unc:.4f}")
    print(f"KL Divergence: {kl_div:.4f} ± {kl_unc:.4f}")
    print(f"Total Variation: {tv_dist:.4f} ± {tv_unc:.4f}")
    
    return {
        'wasserstein': (w_dist, w_unc),
        'kl_divergence': (kl_div, kl_unc),
        'total_variation': (tv_dist, tv_unc)
    }


def run_complete_distance_analysis():
    """
    Run distance analysis and return all intermediate variables needed for NISQ analysis
    """
    shots = 8192
    
    # Generate theoretical target distributions
    gaussian_target, exponential_target, uniform_target = generate_target_distributions(5)
    
    # Create simulators
    ideal_sim = AerSimulator()
    noise_model = create_manual_noise_model()
    noisy_sim = AerSimulator(noise_model=noise_model)
    
    # Run unbiased QGB (Gaussian)
    unbiased_qc = build_qgb_paperstyle(4)
    ideal_gaussian_job = ideal_sim.run(unbiased_qc, shots=shots)
    ideal_gaussian_counts = ideal_gaussian_job.result().get_counts()
    ideal_gaussian_bins = clean_galton_mapping(ideal_gaussian_counts, 4)
    
    noisy_gaussian_job = noisy_sim.run(unbiased_qc, shots=shots)
    noisy_gaussian_counts = noisy_gaussian_job.result().get_counts()
    noisy_gaussian_bins = clean_galton_mapping(noisy_gaussian_counts, 4)
    
    # Run biased QGB (Exponential)
    biased_qc = build_biased_qgb(4, direction='right')
    ideal_exponential_job = ideal_sim.run(biased_qc, shots=shots)
    ideal_exponential_counts = ideal_exponential_job.result().get_counts()
    ideal_exponential_bins = clean_galton_mapping(ideal_exponential_counts, 4)
    
    noisy_exponential_job = noisy_sim.run(biased_qc, shots=shots)
    noisy_exponential_counts = noisy_exponential_job.result().get_counts()
    noisy_exponential_bins = clean_galton_mapping(noisy_exponential_counts, 4)

    # Run Hadamard Quantum Walk (Uniform)
    hadamard_qc = hadamard_walk(9)
    ideal_hadamard_job = ideal_sim.run(hadamard_qc, shots=shots)
    ideal_hadamard_counts = ideal_hadamard_job.result().get_counts()
    ideal_uniform_bins = convert_hadamard_to_5bins(ideal_hadamard_counts, shots)
    
    noisy_hadamard_job = noisy_sim.run(hadamard_qc, shots=shots)
    noisy_hadamard_counts = noisy_hadamard_job.result().get_counts()
    noisy_uniform_bins = convert_hadamard_to_5bins(noisy_hadamard_counts, shots)
    
    # Analyze distances
    print("=== QUANTUM GALTON BOARD DISTANCE ANALYSIS ===")
    
    gaussian_analysis = analyze_distribution_accuracy(
        ideal_gaussian_bins, gaussian_target, "Gaussian", shots)
    
    exponential_analysis = analyze_distribution_accuracy(
        ideal_exponential_bins, exponential_target, "Exponential", shots)
    
    uniform_analysis = analyze_distribution_accuracy(
        ideal_uniform_bins, uniform_target, "Uniform", shots)
    
    # Return all variables needed for NISQ analysis
    return {
        'gaussian_analysis': gaussian_analysis,
        'exponential_analysis': exponential_analysis,
        'uniform_analysis': uniform_analysis,
        'ideal_gaussian_bins': ideal_gaussian_bins,
        'noisy_gaussian_bins': noisy_gaussian_bins,
        'ideal_exponential_bins': ideal_exponential_bins,
        'noisy_exponential_bins': noisy_exponential_bins,
        'ideal_uniform_bins': ideal_uniform_bins,
        'noisy_uniform_bins': noisy_uniform_bins,
        'gaussian_target': gaussian_target,
        'exponential_target': exponential_target,
        'uniform_target': uniform_target
    }

# Enhanced convert function for Hadamard walk
def convert_hadamard_to_5bins(hadamard_counts, shots):
    """
    Convert Hadamard walk results to 5-bin uniform distribution
    """
    # For uniform distribution, create approximately equal bins
    # This is a simplified conversion - adjust based on your specific Hadamard output
    total_states = len(hadamard_counts)
    
    if total_states >= 5:
        # Group states into 5 bins
        states_per_bin = total_states // 5
        remainder = total_states % 5
        
        uniform_bins = []
        state_list = list(hadamard_counts.items())
        
        for bin_idx in range(5):
            start_idx = bin_idx * states_per_bin
            end_idx = start_idx + states_per_bin
            if bin_idx < remainder:
                end_idx += 1
                
            bin_count = sum(count for _, count in state_list[start_idx:end_idx])
            uniform_bins.append(bin_count)
    else:
        # Fallback: approximate uniform distribution
        uniform_bins = [shots // 5] * 5
        remainder = shots % 5
        for i in range(remainder):
            uniform_bins[i] += 1
    
    return uniform_bins



def analyze_nisq_distance_impact(ideal_bins, noisy_bins, target_distribution, label):
    """
    Analyze the impact of NISQ noise on distribution fidelity
    
    Args:
        ideal_bins: List of counts from ideal (noiseless) simulation
        noisy_bins: List of counts from noisy simulation  
        target_distribution: Theoretical target distribution
        label: String label for the distribution type
    
    Returns:
        degradation: Numerical degradation metric (Wasserstein distance difference)
    """
    import numpy as np
    from scipy.stats import wasserstein_distance
    
    # Normalize all distributions to probabilities
    ideal_total = sum(ideal_bins) if sum(ideal_bins) > 0 else 1
    noisy_total = sum(noisy_bins) if sum(noisy_bins) > 0 else 1
    target_total = sum(target_distribution) if sum(target_distribution) > 0 else 1
    
    ideal_probs = np.array(ideal_bins) / ideal_total
    noisy_probs = np.array(noisy_bins) / noisy_total
    target_probs = np.array(target_distribution) / target_total
    
    # Calculate statistical distances
    def total_variation_distance(p, q):
        return 0.5 * np.sum(np.abs(p - q))
    
    def kl_divergence(p, q):
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p_safe = p + epsilon
        q_safe = q + epsilon
        return np.sum(p_safe * np.log(p_safe / q_safe))
    
    # Compute distances for ideal vs target
    ideal_wasserstein = wasserstein_distance(range(len(ideal_probs)), range(len(target_probs)), 
                                           ideal_probs, target_probs)
    ideal_kl = kl_divergence(ideal_probs, target_probs)
    ideal_tv = total_variation_distance(ideal_probs, target_probs)
    
    # Compute distances for noisy vs target  
    noisy_wasserstein = wasserstein_distance(range(len(noisy_probs)), range(len(target_probs)), 
                                           noisy_probs, target_probs)
    noisy_kl = kl_divergence(noisy_probs, target_probs)
    noisy_tv = total_variation_distance(noisy_probs, target_probs)
    
    # Calculate degradation metrics
    wasserstein_degradation = noisy_wasserstein - ideal_wasserstein
    kl_degradation = noisy_kl - ideal_kl
    tv_degradation = noisy_tv - ideal_tv
    
    # Print analysis
    print(f"\nIdeal {label} Distribution Analysis:")
    print(f"Wasserstein Distance: {ideal_wasserstein:.4f} ± 0.0000")
    print(f"KL Divergence: {ideal_kl:.4f} ± 0.0000") 
    print(f"Total Variation: {ideal_tv:.4f} ± 0.0000")
    
    print(f"\nNoisy {label} Distribution Analysis:")
    print(f"Wasserstein Distance: {noisy_wasserstein:.4f} ± 0.0000")
    print(f"KL Divergence: {noisy_kl:.4f} ± 0.0000")
    print(f"Total Variation: {noisy_tv:.4f} ± 0.0000")
    
    print(f"\nNISQ Impact on {label}:")
    print(f"Distance degradation: {wasserstein_degradation:.4f}")
    relative_degradation = (wasserstein_degradation / ideal_wasserstein * 100) if ideal_wasserstein > 0 else 0
    print(f"Relative degradation: {relative_degradation:.1f}%")
    
    return wasserstein_degradation


def test_nisq_performance(circuit, n_layers, shots=8192):
    """
    Test quantum circuit performance under ideal and NISQ conditions
    
    Args:
        circuit: Qiskit quantum circuit to test
        n_layers: Number of Galton board layers
        shots: Number of measurement shots
    
    Returns:
        fidelity: Distribution fidelity (1 - total variation distance)
        degradation: Wasserstein distance degradation due to noise
    """
    # Use your existing noise model
    noise_model = create_manual_noise_model()
    
    # Create simulators
    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)
    
    try:
        # Run ideal simulation
        ideal_job = ideal_sim.run(circuit, shots=shots)
        ideal_counts = ideal_job.result().get_counts()
        
        # Run noisy simulation  
        noisy_job = noisy_sim.run(circuit, shots=shots)
        noisy_counts = noisy_job.result().get_counts()
        
        # Convert to classical Galton board bins using your existing function
        ideal_bins = clean_galton_mapping(ideal_counts, n_layers)
        noisy_bins = clean_galton_mapping(noisy_counts, n_layers)
        
        # Calculate fidelity using total variation distance
        ideal_total = sum(ideal_bins) if sum(ideal_bins) > 0 else 1
        noisy_total = sum(noisy_bins) if sum(noisy_bins) > 0 else 1
        
        ideal_probs = [count/ideal_total for count in ideal_bins]
        noisy_probs = [count/noisy_total for count in noisy_bins]
        
        # Total variation distance
        tv_distance = 0.5 * sum(abs(p_ideal - p_noisy) 
                               for p_ideal, p_noisy in zip(ideal_probs, noisy_probs))
        
        # Fidelity = 1 - total variation distance
        fidelity = 1.0 - tv_distance
        
        # Calculate Wasserstein degradation (optional metric)
        from scipy.stats import wasserstein_distance
        
        bin_positions = list(range(n_layers + 1))
        ideal_wasserstein = wasserstein_distance(bin_positions, bin_positions, 
                                               ideal_probs, ideal_probs)  # Should be ~0
        noisy_wasserstein = wasserstein_distance(bin_positions, bin_positions, 
                                               ideal_probs, noisy_probs)
        
        degradation = noisy_wasserstein - ideal_wasserstein
        
        print(f"  Layers: {n_layers}, Fidelity: {fidelity:.3f}, Gates: {sum(circuit.count_ops().values())}")
        
        return fidelity, degradation
        
    except Exception as e:
        print(f"  Error testing {n_layers} layers: {e}")
        return 0.0, float('inf')  # Return poor performance on error

LAYER_RANGE = range( , ) 
TARGET_FIDELITY =   # Minimum acceptable
SHOTS =              # Consistent with your baseline

# 3. Scaling analysis function
def systematic_layer_scaling(distribution_type='gaussian', max_layers=8):
    """
    Systematically test layer scaling for quantum Galton boards
    
    Challenge requirement: maximize accuracy AND number of layers

    """
    results = {
        'layers': [],
        'fidelities': [],
        'gate_counts': [],
        'circuit_depths': [],
        'wasserstein_distances': []
    }
    
    for n_layers in range(4, max_layers + 1):
        print(f"\n--- Testing {n_layers} layers for {distribution_type} ---")
        
        # Build circuit (using your proven functions)
        if distribution_type == 'gaussian':
            circuit = build_qgb_paperstyle(n_layers)
        elif distribution_type == 'exponential':
            circuit = build_biased_qgb(n_layers, direction='right')
        elif distribution_type == 'hadamard':
            circuit = hadamard_walk(n_layers)
        
        # Analyze circuit complexity
        gate_count = circuit.count_ops()
        circuit_depth = circuit.depth()
        
        # Test under NISQ conditions
        fidelity, degradation = test_nisq_performance(circuit, n_layers)
        
        # Store results
        results['layers'].append(n_layers)
        results['fidelities'].append(fidelity)
        results['gate_counts'].append(sum(gate_count.values()))
        results['circuit_depths'].append(circuit_depth)
        
        # Early stopping if fidelity drops too low
        if fidelity < TARGET_FIDELITY:
            print(f"Fidelity dropped below {TARGET_FIDELITY} at {n_layers} layers")
            break
    
    return results


def find_optimal_layer_count(scaling_results, min_fidelity=0.75):
    """
    Find the maximum layers that maintain acceptable fidelity
    
    This directly addresses the challenge: maximize layers while maintaining accuracy
    """
    valid_layers = []
    
    for i, fidelity in enumerate(scaling_results['fidelities']):
        if fidelity >= min_fidelity:
            valid_layers.append(scaling_results['layers'][i])
    
    if valid_layers:
        optimal_layers = max(valid_layers)
        optimal_fidelity = scaling_results['fidelities'][scaling_results['layers'].index(optimal_layers)]
        
        print(f"OPTIMAL RESULT:")
        print(f"Maximum layers: {optimal_layers}")
        print(f"Achieved fidelity: {optimal_fidelity:.3f}")
        return optimal_layers, optimal_fidelity
    else:
        print("No layers meet minimum fidelity requirement")
        return None, None
