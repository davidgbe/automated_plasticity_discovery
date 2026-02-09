import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pickle
from flax import struct

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

@struct.dataclass
class SimParams:
    """Simulation parameters as a static structure for JAX"""
    n: int
    tau_m: float
    tau_z: float
    tau_x_filt: float  # Time constant for x_ct_1 filtering
    learning_rate: float
    homeo_rate: float
    alpha: float
    presyn_setpoint: float
    dt: float

def gen_gaussian(x, mu, sigma):
    """Generate Gaussian function"""
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

@partial(jit, static_argnames=['params'])
def system_dynamics_step(state, u_t, W0, w_inh, params: SimParams):
    """
    Single step of system dynamics using Euler integration
    
    state: [x, z_filt, x_ct_1_filt, W_flat]
    """
    n = params.n
    W_dim = 3 * n
    
    # Unpack state
    x_in = state[:W_dim]
    z_filt = state[W_dim:W_dim + n]
    x_ct_1_filt = state[W_dim + n:W_dim + 2*n]  # Low-pass filtered x_ct_1
    W = state[W_dim + 2*n:].reshape((W_dim, W_dim))
    
    # Ensure non-negative activity
    x = jnp.clip(x_in, 0, None)
    
    # State dynamics
    dx_dt_raw = (1 / params.tau_m) * ((W - w_inh) @ x - x + u_t)
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
    dz_filt_dt = (z - z_filt) / params.tau_z
    
    # x_ct_1 low-pass filter
    dx_ct_1_filt_dt = (x_ct_1 - x_ct_1_filt) / params.tau_x_filt
    
    # Rectified high-pass signal
    z_hp = jnp.maximum(z - z_filt, 0.0)
    
    comp_to_bound = params.presyn_setpoint - W[:n, :n].sum(axis=0)
    
    # Use filtered x_ct_1 in plasticity rule
    dw_dt_ct_1 = (
        params.learning_rate
        * ((0 * jnp.outer(z_hp * dx_dt[:n], x_ct_1) + jnp.outer(z_hp * x_ct_1, dx_dt[:n]) - 0 * params.alpha * (z * z_hp * x_ct_1)[:, None]) + 1 * params.alpha * jnp.outer(z * z_hp, x_ct_1))  # Changed to x_ct_1_filt
    ) + params.homeo_rate * jnp.where(comp_to_bound > 0, 0, comp_to_bound)[None, :]

    
    # Zero diagonal
    dw_dt_ct_1 = dw_dt_ct_1.at[jnp.diag_indices(n)].set(0)
    
    dw_dt = jnp.zeros((W_dim, W_dim))
    dw_dt = dw_dt.at[:n, :n].set(dw_dt_ct_1)
    
    # Euler integration
    dt = params.dt
    new_x = jnp.clip(x_in + dx_dt * dt, 0, None) 
    new_z_filt = z_filt + dz_filt_dt * dt
    new_x_ct_1_filt = x_ct_1_filt + dx_ct_1_filt_dt * dt
    new_W = W + dw_dt * dt
    new_W_pool = new_W[:n, :n]
    new_W = new_W.at[:n, :n].set(jnp.where(new_W_pool > 0,  new_W_pool, 0))
    
    # Pack new state
    new_state = jnp.concatenate([
        new_x,
        new_z_filt,
        new_x_ct_1_filt,
        new_W.ravel(),
    ])
    
    return new_state, (new_x, new_W, params.alpha * z**2 * x_ct_1, dx_dt[:n] * z_hp)

@partial(jit, static_argnames=['params'])
def simulate_epoch(initial_state, u_trajectory, w_inh, params: SimParams):
    """Simulate one epoch with pre-computed input trajectory"""
    W0 = None  # Not used in step function
    
    def scan_fn(state, u_t):
        new_state, outputs = system_dynamics_step(state, u_t, W0, w_inh, params)
        return new_state, outputs
    
    final_state, (x_history, w_history, z_lead, dx_dt) = jax.lax.scan(scan_fn, initial_state, u_trajectory)
    
    return final_state, x_history, w_history, z_lead, dx_dt

def initialize_weights(n, weight_perturbation, w_e_scale, w_pool_to_shift, w_shift_to_pool, key):
    """Initialize weight matrix"""
    W0 = jnp.zeros((3 * n, 3 * n))
    
    # # Pool-to-pool connections
    # shift_mats_pool_pool = []
    # for i in range(0, n):
    #     w_shift = (
    #         jnp.diag(jnp.ones(n - jnp.abs(i)), k=i)
    #         * 0.5
    #         * (1 + jnp.cos(2 * jnp.pi * jnp.abs(i) / (n+1)))
    #     )
    #     shift_mats_pool_pool.append(w_shift)
    #     shift_mats_pool_pool.append(w_shift.T)
    
    # weight_pert = jax.random.normal(key, (n, n)) * weight_perturbation + 1
    # W0 = W0.at[:n, :n].set(
    #     w_e_scale * jnp.sum(jnp.stack(shift_mats_pool_pool), axis=0) * weight_pert
    # )

    weight_pert = jax.random.uniform(key, (n, n)) * weight_perturbation

    W0 = W0.at[:n, :n].set(jnp.array([
        [1.5, weight_pert[0, 1]],
        [weight_pert[1, 0], 1.8],
    ]))
    
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
    W0 = W0.at[n:2*n, n:2*n].set(-1)
    W0 = W0.at[2*n:3*n, 2*n:3*n].set(-1)
    
    return W0

def make_u_trajectory(n, t, dt, key):
    """Create input trajectory for one epoch"""
    block_len = 0.05
    inp = np.zeros((3 + n, len(t)))
    
    for t_p in np.linspace(0, t.max(), int((t.max() + 1)/block_len)):
        if t_p >= 0.1 and t_p < 0.5:
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
    learning_rate=0,
    homeo_rate=0,
    alpha=10,
    presyn_setpoint=3.5,
    tau_x_filt=0.02,  # Time constant for x_ct_1 filtering
    tau_z=2.5e-3,
    w_e_scale=0.864,
    w_pool_to_shift=0.5,
    w_shift_to_pool=0.25,
    weight_perturbation=0,
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
    
    # Setup parameters using NamedTuple
    t = np.arange(t_sim[0], t_sim[1], dt)
    params = SimParams(
        n=n,
        tau_m=1e-2,
        tau_z=tau_z,
        tau_x_filt=tau_x_filt,
        learning_rate=learning_rate,
        homeo_rate=homeo_rate,
        alpha=alpha,
        presyn_setpoint=presyn_setpoint,
        dt=dt,
    )
    
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
        w_inh_vec = jnp.array(np.random.normal(size=(n,), loc=1, scale=0) * 7)
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
        x_ct_1_filt0 = jnp.zeros(n)  # Initialize filtered x_ct_1
        state = jnp.concatenate([x_init, z_filt0, x_ct_1_filt0, W0.ravel()])
        
        # Storage for this network
        weight_trajectory = [W0.copy()]  # Store initial weights
        last_20_epochs_data = []
        z_lead = []
        dx_dt = []
        
        start_time = time()
        
        for epoch in range(n_epochs):
            # Generate new input for this epoch
            u_trajectory, inp = make_u_trajectory(n, t, dt, key)
            key, _ = jax.random.split(key)
            
            # Simulate epoch
            state, x_history, w_history, z_lead_history, dx_dt_history = simulate_epoch(state, u_trajectory, w_inh, params)
            
            # Store final weights
            final_W = state[3*n + 2*n:].reshape((3*n, 3*n))
            weight_trajectory.append(np.array(final_W))
            z_lead.append(z_lead_history)
            dx_dt.append(dx_dt_history)

            
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

            state = state.at[:3*n].set(x_init)
            state = state.at[3*n:3*n + n].set(z_filt0)
            state = state.at[3*n + n:3*n + 2*n].set(x_ct_1_filt0)
        
        all_results.append({
            'weight_trajectory': weight_trajectory,
            'last_20_epochs': last_20_epochs_data,
            'final_weights': weight_trajectory[-1],
        })
    
    return all_results, t


# Example usage
if __name__ == "__main__":
    # Train networks
    results, t = train_multiple_networks(
        n_networks=5,
        n_epochs=5000,
        n=2,
        t_sim=(0, 1.5),
        dt=1e-4,
        learning_rate=2e4,
        homeo_rate=0,
        alpha=37, #1,
        presyn_setpoint=6,
        tau_x_filt=0.005,  # Time constant for x_ct_1 filtering
        tau_z=2.5e-3,
        w_e_scale=2, #0.864,
        w_pool_to_shift=0.5,
        w_shift_to_pool=0.05,
        weight_perturbation=1.0,
        peak_amp=0.5,
        seed=100,
    )

    # Save results
    with open('network_training_results_w_change_w_ii.pkl', 'wb') as f:
        pickle.dump({'results': results, 't': t}, f)
    
    # Plot results
    # fig_w, fig_w_traj = plot_results(results, t, n=10)
    
    print("\nTraining complete! Results saved.")
