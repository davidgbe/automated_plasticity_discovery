from run_active_tuning_handbuilt import train_multiple_networks
import numpy as np
from time import time
import pickle

# Hyperparameter sweep for alpha and tau_z
if __name__ == "__main__":
    # Define parameter grids
    alpha_values = np.linspace(25, 50, 25)  # 1 to 100
    tau_z_values = np.linspace(2.5e-3, 2.5e-3, 1)  # 0.001 to 0.01
    
    # Fixed parameters
    fixed_params = {
        'n_networks': 5,
        'n_epochs': 3000,
        'n': 2,
        't_sim': (0, 1.5),
        'dt': 1e-4,
        'learning_rate': 800,
        'homeo_rate': 0,
        'presyn_setpoint': 6,
        'tau_x_filt': 0.005,
        'w_e_scale': 2,
        'w_pool_to_shift': 0.5,
        'w_shift_to_pool': 0.05,
        'weight_perturbation': 0.1,
        'peak_amp': 0.5,
        'seed': 81,
    }
    
    # Storage for results
    correlation_matrix = np.zeros((len(alpha_values), len(tau_z_values)))
    
    total_runs = len(alpha_values) * len(tau_z_values)
    current_run = 0
    
    print(f"Starting hyperparameter sweep: {len(alpha_values)} alpha × {len(tau_z_values)} tau_z = {total_runs} total runs")
    print(f"Alpha range: {alpha_values.min():.3f} to {alpha_values.max():.3f}")
    print(f"Tau_z range: {tau_z_values.min():.4f} to {tau_z_values.max():.4f}")
    
    start_time_total = time()
    
    for i, alpha in enumerate(alpha_values):
        for j, tau_z in enumerate(tau_z_values):
            current_run += 1
            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs}: alpha={alpha:.2f}, tau_z={tau_z:.4f}")
            print(f"{'='*60}")
            
            run_start = time()
            
            # Train networks with current parameters
            results, t = train_multiple_networks(
                alpha=alpha,
                tau_z=tau_z,
                **fixed_params
            )
            
            # Compute correlations for each network
            dt = fixed_params['dt']
            n = fixed_params['n']
            input_start_int = int(0.5 / dt)  # Skip initial transient
            
            network_correlations = []
            
            for net_idx, result in enumerate(results):
                ending_locs = []
                int_values = []
                
                for epoch_data in result['last_20_epochs']:
                    x_history = epoch_data['x_history']
                    color_val = epoch_data['color_val']
                    
                    # Compute bump location
                    x_pool = x_history[:, :n]
                    x_sum = x_pool.sum(axis=1, keepdims=True) + 1e-6
                    x_normalized = x_pool / x_sum
                    loc_trace = (x_normalized * np.arange(n)).sum(axis=1)
                    loc_trace = np.where(x_pool.sum(axis=1) > 1e-6, loc_trace, np.nan)
                    
                    # Sample a random timepoint after input starts
                    rand_index = np.random.randint(loc_trace.shape[0] - input_start_int) + input_start_int
                    ending_locs.append(loc_trace[rand_index])
                    int_values.append(color_val)
                
                # Compute correlation
                corr = np.corrcoef(ending_locs, int_values)[0, 1]
                network_correlations.append(corr)
                print(f"  Network {net_idx+1} correlation: {corr:.4f}")
            
            # Store mean correlation across networks
            mean_corr = np.mean(network_correlations)
            correlation_matrix[i, j] = mean_corr
            
            run_time = time() - run_start
            elapsed_total = time() - start_time_total
            estimated_remaining = (elapsed_total / current_run) * (total_runs - current_run)
            
            print(f"\nMean correlation: {mean_corr:.4f}")
            print(f"Run time: {run_time:.1f}s")
            print(f"Elapsed: {elapsed_total/60:.1f}min, Est. remaining: {estimated_remaining/60:.1f}min")
    
    # Save results
    sweep_results = {
        'correlation_matrix': correlation_matrix,
        'alpha_values': alpha_values,
        'tau_z_values': tau_z_values,
        'fixed_params': fixed_params,
    }
    
    with open('hyperparam_sweep_results.pkl', 'wb') as f:
        pickle.dump(sweep_results, f)
