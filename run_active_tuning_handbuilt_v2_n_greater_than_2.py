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
    n: int
    tau_m: float
    learning_rate: float
    homeo_rate: float
    alpha: float
    presyn_setpoint: float
    dt: float

    # Plasticity coefficients
    hebbian_dx_scale: float = 200.0
    hebbian_dx_scale_conj: float = 200.0
    alpha_outer_scale: float = 1.5

    # Time constants for traces
    tau_x_trace_1: float = 0.02
    tau_z_trace_1: float = 0.02
    tau_x_trace_2: float = 0.02
    tau_z_trace_2: float = 0.02
    tau_x_trace_3: float = 0.02  # for outer(x_tilde_3, x_ct_1)
    tau_x_trace_4: float = 0.02  # for outer(x_ct_1, x_tilde_4)

    # New term scales
    outer_zx_x_scale: float = 0.0
    outer_xtilde_x_scale: float = 0.0
    outer_x_xtilde_scale: float = 0.0


def gen_gaussian(x, mu, sigma):
    """Generate Gaussian function"""
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


@partial(jit, static_argnames=['params'])
def system_dynamics_step(state, u_t, W0, w_inh, params: SimParams):
    """
    Single step of system dynamics using Euler integration

    state: [x, trace_state, W_flat]
    trace_state layout (6*n):
        [x_tilde_1 | z_tilde_1 | x_tilde_2 | z_tilde_2 | x_tilde_3 | x_tilde_4]
    """
    n = params.n
    W_dim = 3 * n
    trace_dim = 6 * n

    # Unpack state
    x_in = state[:W_dim]
    trace_state = state[W_dim:W_dim + trace_dim]
    W = state[W_dim + trace_dim:].reshape((W_dim, W_dim))

    x_tilde_1 = trace_state[:n]
    z_tilde_1 = trace_state[n:2*n]
    x_tilde_2 = trace_state[2*n:3*n]
    z_tilde_2 = trace_state[3*n:4*n]
    x_tilde_3 = trace_state[4*n:5*n]
    x_tilde_4 = trace_state[5*n:6*n]

    # Ensure non-negative activity
    x = jnp.clip(x_in, 0, None)

    # State dynamics
    dx_dt_raw = (1 / params.tau_m) * ((W - w_inh) @ x - x + u_t)
    dx_dt = jnp.where(
        jnp.logical_and(x <= 0, dx_dt_raw < 0),
        0,
        dx_dt_raw,
    )

    # Subpopulations
    x_ct_1 = x[:n]
    x_ct_2 = x[n:3 * n]
    z = W[:n, n:3 * n] @ x_ct_2

    # Trace dynamics
    dx_tilde_1_dt = (-x_tilde_1 + x_ct_1) / params.tau_x_trace_1
    dz_tilde_1_dt = (-z_tilde_1 + z) / params.tau_z_trace_1

    dx_tilde_2_dt = (-x_tilde_2 + x_ct_1) / params.tau_x_trace_2
    dz_tilde_2_dt = (-z_tilde_2 + z) / params.tau_z_trace_2

    dx_tilde_3_dt = (-x_tilde_3 + x_ct_1) / params.tau_x_trace_3
    dx_tilde_4_dt = (-x_tilde_4 + x_ct_1) / params.tau_x_trace_4

    comp_to_bound = params.presyn_setpoint - W[:n, :n].sum(axis=0)

    # Plasticity rule
    dw_dt_ct_1 = (
        params.learning_rate
        * (
            params.hebbian_dx_scale       * jnp.outer(z_tilde_1 * x_ct_1, x_tilde_1)
            + params.hebbian_dx_scale_conj * jnp.outer(z_tilde_2 * x_tilde_2, x_ct_1)
            - params.alpha                 * (z * x_ct_1)[:, None]
            + params.alpha_outer_scale * params.alpha * jnp.outer(z, x_ct_1)
            + params.outer_zx_x_scale     * jnp.outer(z * x_ct_1, x_ct_1)
            + params.outer_xtilde_x_scale * jnp.outer(x_tilde_3, x_ct_1)
            + params.outer_x_xtilde_scale * jnp.outer(x_ct_1, x_tilde_4)
        )
    ) + params.homeo_rate * jnp.where(comp_to_bound > 0, 0, comp_to_bound)[None, :]

    # Zero diagonal
    dw_dt_ct_1 = dw_dt_ct_1.at[jnp.diag_indices(n)].set(0)

    dw_dt = jnp.zeros((W_dim, W_dim))
    dw_dt = dw_dt.at[:n, :n].set(dw_dt_ct_1)

    # Euler integration
    dt = params.dt
    new_x = jnp.clip(x_in + dx_dt * dt, 0, None)

    new_x_tilde_1 = x_tilde_1 + dx_tilde_1_dt * dt
    new_z_tilde_1 = z_tilde_1 + dz_tilde_1_dt * dt
    new_x_tilde_2 = x_tilde_2 + dx_tilde_2_dt * dt
    new_z_tilde_2 = z_tilde_2 + dz_tilde_2_dt * dt
    new_x_tilde_3 = x_tilde_3 + dx_tilde_3_dt * dt
    new_x_tilde_4 = x_tilde_4 + dx_tilde_4_dt * dt

    new_W = W + dw_dt * dt
    new_W_pool = new_W[:n, :n]
    new_W = new_W.at[:n, :n].set(jnp.where(new_W_pool > 0, new_W_pool, 0))

    new_trace_state = jnp.concatenate([
        new_x_tilde_1,
        new_z_tilde_1,
        new_x_tilde_2,
        new_z_tilde_2,
        new_x_tilde_3,
        new_x_tilde_4,
    ])

    new_state = jnp.concatenate([
        new_x,
        new_trace_state,
        new_W.ravel(),
    ])

    return new_state, (new_x, new_W)


@partial(jit, static_argnames=['params'])
def simulate_epoch(initial_state, u_trajectory, w_inh, params: SimParams):
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
    for i in range(0, n):
        w_shift = (
            jnp.diag(jnp.ones(n - jnp.abs(i)), k=i)
            * 0.5
            * (1 + jnp.cos(2 * jnp.pi * jnp.abs(i) / (n+1)))
        )
        shift_mats_pool_pool.append(w_shift)
        shift_mats_pool_pool.append(w_shift.T)

    weight_pert = jax.random.normal(key, (n, n)) * weight_perturbation * 0 + 1
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
    W0 = W0.at[n:2*n, n:2*n].set(-1)
    W0 = W0.at[2*n:3*n, 2*n:3*n].set(-1)

    return W0


def make_u_trajectory(n, t, dt, key):
    """Create input trajectory for one epoch"""
    block_len = 0.05
    inp = np.zeros((3 + n, len(t)))

    for t_p in np.linspace(0, t.max(), int((t.max() + 1)/block_len)):
        if t_p >= 0.1 and t_p < 1.5:
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
    t_sim=(0, 1.5),
    dt=0.002,
    learning_rate=0,
    homeo_rate=0,
    alpha=10,
    alpha_outer_scale=1.5,
    hebbian_dx_scale=200,
    hebbian_dx_scale_conj=0,
    tau_x_trace_1=0.02,
    tau_z_trace_1=0.02,
    tau_x_trace_2=0.02,
    tau_z_trace_2=0.02,
    tau_x_trace_3=0.02,
    tau_x_trace_4=0.02,
    presyn_setpoint=3.5,
    w_e_scale=0.864,
    w_pool_to_shift=0.5,
    w_shift_to_pool=0.25,
    weight_perturbation=0,
    peak_amp=0.5,
    outer_zx_x_scale=0.0,
    outer_xtilde_x_scale=0.0,
    outer_x_xtilde_scale=0.0,
    seed=42,
):
    """
    Train multiple networks over epochs

    Returns:
        results: dict containing weight trajectories and activities
    """
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    t = np.arange(t_sim[0], t_sim[1], dt)
    trace_dim = 6 * n

    params = SimParams(
        n=n,
        tau_m=1e-2,
        learning_rate=learning_rate,
        homeo_rate=homeo_rate,
        alpha=alpha,
        alpha_outer_scale=alpha_outer_scale,
        hebbian_dx_scale=hebbian_dx_scale,
        hebbian_dx_scale_conj=hebbian_dx_scale_conj,
        tau_x_trace_1=tau_x_trace_1,
        tau_z_trace_1=tau_z_trace_1,
        tau_x_trace_2=tau_x_trace_2,
        tau_z_trace_2=tau_z_trace_2,
        tau_x_trace_3=tau_x_trace_3,
        tau_x_trace_4=tau_x_trace_4,
        presyn_setpoint=presyn_setpoint,
        outer_zx_x_scale=outer_zx_x_scale,
        outer_xtilde_x_scale=outer_xtilde_x_scale,
        outer_x_xtilde_scale=outer_x_xtilde_scale,
        dt=dt,
    )

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
        trace_init = jnp.zeros(trace_dim)
        state = jnp.concatenate([
            x_init,
            trace_init,
            W0.ravel()
        ])

        # Storage for this network
        weight_trajectory = [W0.copy()]
        last_20_epochs_data = []

        start_time = time()

        for epoch in range(n_epochs):
            # Generate new input for this epoch
            u_trajectory, inp = make_u_trajectory(n, t, dt, key)
            key, _ = jax.random.split(key)

            # Simulate epoch
            state, x_history, w_history = simulate_epoch(state, u_trajectory, w_inh, params)

            # Store final weights
            final_W = state[3*n + trace_dim:].reshape((3*n, 3*n))
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
                pass

            # Reset activity and traces, keep weights
            state = state.at[:3*n + trace_dim].set(
                jnp.concatenate([
                    x_init,
                    jnp.zeros(trace_dim)
                ])
            )

        all_results.append({
            'weight_trajectory': weight_trajectory,
            'last_20_epochs': last_20_epochs_data,
            'final_weights': weight_trajectory[-1],
        })

    return all_results, t


if __name__ == "__main__":
    results, t = train_multiple_networks(
        n_networks=1,
        n_epochs=10000,
        n=5,
        t_sim=(0, 2.0),
        dt=1e-4,
        learning_rate=10.65,
        alpha=20.21,
        alpha_outer_scale=2.11,
        hebbian_dx_scale=9.49,
        hebbian_dx_scale_conj=4.67,
        homeo_rate=1.33,
        presyn_setpoint=10.62,
        tau_x_trace_1=22.1 * 1e-3,
        tau_z_trace_1=19.3 * 1e-3,
        tau_x_trace_2=19.2 * 1e-3,
        tau_z_trace_2=18.4 * 1e-3,
        tau_x_trace_3=0.02,
        tau_x_trace_4=0.02,
        outer_zx_x_scale=0.0,
        outer_xtilde_x_scale=0.0,
        outer_x_xtilde_scale=0.0,
        w_e_scale=2,
        w_pool_to_shift=0.5,
        w_shift_to_pool=0.3,
        weight_perturbation=1.0,
        peak_amp=0.5,
        seed=0,
    )

    with open('network_training_results_v2_rule_n5.pkl', 'wb') as f:
        pickle.dump({'results': results, 't': t}, f)

    print("\nTraining complete! Results saved.")