import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pickle

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def gen_gaussian(x, mu, sigma):
    """Generate Gaussian function"""
    return jnp.exp(-0.5 * ((x - mu)/sigma) ** 2) / (2 * jnp.pi * sigma ** 2)

@jit
def system_dynamics_step(state, u_t, W0, w_inh, params):
    """
    Single step of system dynamics using Euler integration
    
    state: [x, z_filt, W_flat]
    """
    n = params['n']
    W_dim = 3 * n
    
    # Unpack state
    x_in = state[:W_dim]
    z_filt = state[W_dim:W_dim + n]
    W = state[W_dim + n:].reshape((W_dim, W_dim))
    
    # Ensure non-negative activity
    x = jnp.clip(x_in, 0, None)
    
    # State dynamics
    dx_dt_raw = (1 / params['tau_m']) * ((W - w_inh) @ x - x + u_t)
    dx_dt = jnp.where(
        jnp.logical_and(x <= 0, dx_dt_raw < 0),
        0,
        dx_dt_raw,
    )
    
    # Weight dynamics
    x_ct_1 = x[:n]
    x_ct_2 = x[n:3 * n]
    
    z = W[:n, n:3 * n] @ x_ct_2
    
    # z low-pass filter
    dz_filt_dt = (z - z_filt) / params['tau_z']
    
    # Rectified high-pass signal
    z_hp = jnp.maximum(z - z_filt, 0.0)
    
    comp_to_bound = params['presyn_setpoint'] - W[:n, :n].sum(axis=0)
    
    dw_dt_ct_1 = (
        params['learning_rate']
        * z_hp
        * jnp.outer(params['alpha'] * z - dx_dt[:n], x_ct_1)
    ) + params['homeo_rate'] * jnp.where(comp_to_bound > 0, 0, comp_to_bound)[None, :]
    
    # Zero diagonal
    dw_dt_ct_1 = dw_dt_ct_1.at[jnp.diag_indices(n)].set(0)
    
    dw_dt = jnp.zeros((W_dim, W_dim))
    dw_dt = dw_dt.at[:n, :n].set(dw_dt_ct_1)
    
    # Euler integration
    dt = params['dt']
    new_x = x_in + dx_dt * dt
    new_z_filt = z_filt + dz_filt_dt * dt
    new_W = W + dw_dt * dt
    
    # Pack new state
    new_state = jnp.concatenate([
        new_x,
        new_z_filt,
        new_W.ravel(),
    ])
    
    return new_state, (new_x, new_W)

@partial(jit, static_argnums=(3,))
def simulate_epoch(initial_state, u_trajectory, w_inh, params):
    """Simulate one epoch with pre-computed input trajectory"""
    W0 = None  # Not used in step function
    
    def scan_fn(state, u_t):
        new_state, outputs = system_dynamics_step(state, u_t, W0, w_inh, params)
        return new_state, outputs
    
    final_state, (x_history, w_history) = jax.lax.scan(scan_fn, initial_state, u_trajectory)
    
    return final_state, x_history, w_history

def initialize_weights(n, weight_perturbation, w_e_scale, w_pool_to_shift, w_shift_to_pool, key):
    """Initialize weight matrix"""
    W0 = jnp.zeros((3 * n, 3 * n))
    
    # Pool-to-pool connections
    shift_mats_pool_pool = []
    for i in range(1, n):
        w_shift = (
            jnp.diag(jnp.ones(n - jnp.abs(i)), k=i)
            * 0.5
            * (1 + jnp.cos(2 * jnp.pi * jnp.abs(i) / n))
        )
        shift_mats_pool_pool.append(w_shift)
        shift_mats_pool_pool.append(w_shift.T)
    
    weight_pert = jax.random.normal(key, (n, n)) * weight_perturbation + 1
    W0 = W0.at[:n, :n].set(
        w_e_scale * jnp.sum(jnp.stack(shift_mats_pool_pool), axis=0) * weight_pert
    )
    
    # Shift connections
    W0 = W0.at[:n, n:2*n].set(w_shift_to_pool * jnp.diag(jnp.ones(n-1), k=1))
    W0 = W0.at[:n, 2*n:3*n].set(w_shift_to_pool * jnp.diag(jnp.ones(n-1), k=-1))
    
    # Pool-to-shift connections
    w_side_pool = (
        jnp.diag(jnp.ones(n)) + 
        0.5 * (jnp.diag(jnp.ones(n-1), k=1) + jnp.diag(jnp.ones(n-1), k=-1)) - 1
    ) * w_pool_to_shift
    W0 = W0.at[n:2*n, :n].set(w_side_pool)
    W0 = W0.at[2*n:3*n, :n].set(w_side_pool)
    
    return W0

def make_u_trajectory(n, t, dt, key):
    """Create input trajectory for one epoch"""
    block_len = 0.05
    inp = np.zeros((3 + n, len(t)))
    
    for t_p in np.linspace(0, t.max(), int((t.max() + 1)/block_len)):
        if t_p >= 0.05 and t_p < 1:
            u_block = 0.15 * np.random.rand(2) + 0.05
            if np.random.rand() > 0.5:
                u_block[0] = 0
            else:
                u_block[1] = 0
        else:
            u_block = np.zeros(2)
        inp[:2, int(t_p/dt):int((t_p + block_len)/dt)] = u_block[:, None]
    
    inp[2:2+n, :] = 0.5
    
    # Convert to JAX format: (time, features)
    u_trajectory = jnp.zeros((len(t), 3*n))
    u_trajectory = u_trajectory.at[:, :n].set(inp[2:2+n, :].T)
    u_trajectory = u_trajectory.at[:, n:2*n].set(jnp.repeat(inp[0:1, :].T, n, axis=1))
    u_trajectory = u_trajectory.at[:, 2*n:3*n].set(jnp.repeat(inp[1:2, :].T, n, axis=1))
    
    return u_trajectory, inp

def train_multiple_networks(
    n_networks=5,
    n_epochs=100,
    n=10,
    t_sim=(0, 1.2),
    dt=0.002,
    learning_rate=1000,
    homeo_rate=0,
    alpha=10,
    presyn_setpoint=3.5,
    w_e_scale=0.864,
    w_pool_to_shift=0.5,
    w_shift_to_pool=0.25,
    weight_perturbation=0.05,
    peak_amp=0.5,
    seed=42,
):
    """
    Train multiple networks over epochs
    
    Returns:
        results: dict containing weight trajectories and activities
    """
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    
    # Setup parameters
    t = np.arange(t_sim[0], t_sim[1], dt)
    params = {
        'n': n,
        'tau_m': 1e-2,
        'tau_z': 0.02,
        'learning_rate': learning_rate,
        'homeo_rate': homeo_rate,
        'alpha': alpha,
        'presyn_setpoint': presyn_setpoint,
        'dt': dt,
    }
    
    # Initialize storage
    all_results = []
    
    for net_idx in range(n_networks):
        print(f"\nTraining network {net_idx + 1}/{n_networks}")
        
        # Initialize network
        key, subkey = jax.random.split(key)
        W0 = initialize_weights(n, weight_perturbation, w_e_scale, 
                               w_pool_to_shift, w_shift_to_pool, subkey)
        
        # Initialize inhibition
        w_inh = jnp.zeros((3*n, 3*n))
        w_inh_vec = jnp.array(np.random.normal(size=(n,), loc=1, scale=0) * 2)
        w_inh = w_inh.at[:n, :n].set(w_inh_vec[:, None])
        
        # Initialize state
        x = np.arange(n)
        x_init_peak_loc = 0.5
        s = 1.5
        x_init = jnp.concatenate([
            jnp.clip(peak_amp * gen_gaussian(x, x_init_peak_loc * (n-1), s), 0, None),
            jnp.zeros(2*n),
        ])
        z_filt0 = jnp.zeros(n)
        state = jnp.concatenate([x_init, z_filt0, W0.ravel()])
        
        # Storage for this network
        weight_trajectory = [W0.copy()]  # Store initial weights
        last_20_epochs_data = []
        
        start_time = time()
        
        for epoch in range(n_epochs):
            # Generate new input for this epoch
            u_trajectory, inp = make_u_trajectory(n, t, dt, key)
            key, _ = jax.random.split(key)
            
            # Simulate epoch
            state, x_history, w_history = simulate_epoch(state, u_trajectory, w_inh, params)
            
            # Store final weights
            final_W = state[3*n + n:].reshape((3*n, 3*n))
            weight_trajectory.append(np.array(final_W))
            
            # Store last 20 epochs' activity
            if epoch >= n_epochs - 20:
                color_val = (inp[1].sum() - inp[0].sum())
                last_20_epochs_data.append({
                    'x_history': np.array(x_history),
                    'inp': inp,
                    'color_val': color_val,
                })
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs} - {time() - start_time:.2f}s")
        
        all_results.append({
            'weight_trajectory': weight_trajectory,
            'last_20_epochs': last_20_epochs_data,
            'final_weights': weight_trajectory[-1],
        })
    
    return all_results, t

def plot_results(all_results, t, n, save_path='network_training'):
    """Plot results for all networks"""
    n_networks = len(all_results)
    
    # Plot 1: Final weight matrices for all networks
    fig_w, axes_w = plt.subplots(1, n_networks, figsize=(4*n_networks, 4))
    if n_networks == 1:
        axes_w = [axes_w]
    
    w_max = max(np.abs(result['final_weights']).max() for result in all_results) * 1.05
    
    for i, result in enumerate(all_results):
        im = axes_w[i].matshow(result['final_weights'], cmap='bwr', vmin=-w_max, vmax=w_max)
        axes_w[i].set_title(f'Network {i+1} Final Weights')
        axes_w[i].set_xlabel('Presyn index')
        axes_w[i].set_ylabel('Postsyn index')
        plt.colorbar(im, ax=axes_w[i])
    
    fig_w.tight_layout()
    
    # Plot 2: Weight change trajectories for all networks
    fig_w_traj, axes_w_traj = plt.subplots(1, n_networks, figsize=(4*n_networks, 4))
    if n_networks == 1:
        axes_w_traj = [axes_w_traj]
    
    for i, result in enumerate(all_results):
        w_traj = np.array(result['weight_trajectory'])
        n_epochs = len(w_traj) - 1
        
        # Plot all weight elements over time
        for ii in range(n):
            for jj in range(n):
                axes_w_traj[i].plot(np.arange(n_epochs + 1), w_traj[:, ii, jj], alpha=0.3)
        
        axes_w_traj[i].set_title(f'Network {i+1} Weight Dynamics')
        axes_w_traj[i].set_xlabel('Epoch')
        axes_w_traj[i].set_ylabel('Weight value')
    
    fig_w_traj.tight_layout()
    
    # Plot 3: Bump locations for last 20 epochs (one plot per network)
    for net_idx, result in enumerate(all_results):
        fig_bump, ax_bump = plt.subplots(1, 1, figsize=(8, 6))
        
        # Get color range
        color_vals = [epoch['color_val'] for epoch in result['last_20_epochs']]
        m = max(abs(min(color_vals)), abs(max(color_vals)))
        
        cmap = sns.color_palette('magma', as_cmap=True)
        
        for epoch_data in result['last_20_epochs']:
            x_history = epoch_data['x_history']
            color_val = epoch_data['color_val']
            
            # Compute bump location
            x_pool = x_history[:, :n]
            x_sum = x_pool.sum(axis=1, keepdims=True) + 1e-6
            loc_trace = np.dot(np.arange(n), (x_pool / x_sum).T)
            loc_trace = np.where(x_pool.sum(axis=1) > 1e-3, loc_trace, np.nan)
            
            color = cmap(0.5 * (1 + color_val / m))
            ax_bump.plot(t, loc_trace, color=color, linewidth=1.5, alpha=0.8)
        
        ax_bump.set_xlabel('Time (s)')
        ax_bump.set_ylabel('Bump location')
        ax_bump.set_title(f'Network {net_idx+1} - Last 20 Epochs')
        ax_bump.set_ylim(0, n)
        
        fig_bump.tight_layout()
    
    plt.show()
    
    return fig_w, fig_w_traj

# Example usage
if __name__ == "__main__":
    # Train networks
    results, t = train_multiple_networks(
        n_networks=1,
        n_epochs=50,
        n=10,
        t_sim=(0, 1.2),
        dt=0.002,
        learning_rate=1000,
        homeo_rate=0,
        alpha=10,
        presyn_setpoint=3.5,
        w_e_scale=0.864,
        w_pool_to_shift=0.5,
        w_shift_to_pool=0.25,
        weight_perturbation=0.05,
        peak_amp=0.5,
        seed=42,
    )

    # Save results
    with open('network_training_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 't': t}, f)
    
    # Plot results
    # fig_w, fig_w_traj = plot_results(results, t, n=10)
    
    print("\nTraining complete! Results saved.")