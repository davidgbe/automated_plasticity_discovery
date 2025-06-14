import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import diffrax
import jax.random as jr
from aux_funcs import merge_with_indices_jax
from jax import lax

R_RESCALING = 10
R_EXP_RESCALING = 10
W_RESCALING = 10
ALPHA = 1000
BETA = 1/ALPHA
SOFTPLUS_TRANSITION = 1e-3

@jax.jit
def _delta_W_ij_two_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j):
    return jnp.array(
        [
            jnp.float32(1),
            r_i,
            r_j,
            r_i * r_j,
            r_exp_i[0],
            r_exp_i[1] * r_i,
            r_exp_i[2] * r_j,
            r_exp_j[3],
            r_exp_j[4] * r_j,
            r_exp_j[5] * r_i,
            w_ij,
            w_ij * r_i,
            w_ij * r_j,
            w_ij * r_i * r_j,
            w_ij * r_exp_i[6],
            w_ij * r_exp_i[7] * r_i,
            w_ij * r_exp_i[8] * r_j,
            w_ij * r_exp_j[9],
            w_ij * r_exp_j[10] * r_j,
            w_ij * r_exp_j[11] * r_i,
        ]
    )


@jax.jit
def _delta_W_ij_two_factor(w_ij, r_i, r_j, r_exp_i, r_exp_j, c):
    delta_w =  c * _delta_W_ij_two_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j)
    return delta_w.sum(), jnp.abs(delta_w).sum()


delta_W_ij_two_factor = jax.vmap(
    jax.vmap(_delta_W_ij_two_factor, (0, None, 0, None, 0, None)),
    (0, 0, None, 0, None, None),
)


@jax.jit
def _delta_W_ij_three_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i):
    return jnp.array([
        r_exp_i[0] * f_i[0],
        r_exp_i[1] * r_j * f_i[1],
        r_exp_j[2] * f_i[2],
        r_exp_j[3] * r_i * f_i[3],
        w_ij * r_exp_i[4] * f_i[4],
        w_ij * r_exp_i[5] * r_j * f_i[5],
        w_ij * r_exp_j[6] * f_i[6],
        w_ij * r_exp_j[7] * r_i * f_i[7],
    ])


@jax.jit
def _delta_W_ij_three_factor(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i, c):
    delta_w = c * _delta_W_ij_three_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i)
    return delta_w.sum(), jnp.abs(delta_w).sum()


delta_W_ij_three_factor = jax.vmap(
    jax.vmap(_delta_W_ij_three_factor, (0, None, 0, None, 0, None, None)),
    (0, 0, None, 0, None, 0, None),
)


@jax.jit
def _softplus(a):
    return jnp.where(
        a > SOFTPLUS_TRANSITION,
        a,
        1/ALPHA * jnp.log(1 + jnp.exp(a/BETA))
    )


softplus = jax.vmap(
    jax.vmap(_softplus, (0,)),
    (0,),
)


def inv_softplus(w):
    return jnp.where(
        jnp.abs(w) < SOFTPLUS_TRANSITION,
        BETA * jnp.log(jnp.exp(ALPHA * jnp.abs(w)) - 1), 
        jnp.abs(w),
    )


# @partial(jax.jit, static_argnames=['s_offsets', 'g', 'n_e', 'n_i'])
def calc_r_from_s(s, s_offsets, g, n_e, n_i):
    s_thresh = jnp.maximum(s - s_offsets, 0)

    e_act = lax.dynamic_slice(s_thresh, (0,), (n_e,))
    i_act = lax.dynamic_slice(s_thresh, (n_e,), (n_i,))

    r = g * jnp.concatenate((jnp.tanh(e_act), i_act)) # excitatory cells get a tanh threshold, inhibition is left as threshold linear
    return r


# @partial(jax.jit, static_argnames=['g', 's_offsets', 'w_u', ' tau_s', 'eta', 'n_e', 'n_i', 'n_e_pool', 'n_e_side'])
def _learning_dynamics(t, y, r_in, w_polarity, w_nonzero, c, tau_rules, time, g, s_offsets, w_u, tau_s, eta, n_e, n_i, n_e_pool, n_e_side):

    def u(t_prime):
        return jax.vmap(jnp.interp, (None, None, 1),)(t_prime, time, r_in)

    n_1 = n_e_pool
    n_2 = 2 * n_e_side
    n_plastic = n_1 + n_2

    s, r_exp, a, syn, unstable = y

    W = w_polarity * softplus(a) * w_nonzero
    unstable_bool = jnp.any(unstable > 0)

    delta_unstable = jnp.any(a > 20) | jnp.any(s > 10) | unstable_bool
    mask = ~(delta_unstable | unstable_bool)

    # print('mask')
    # jax.debug.print('{}', mask)
    # jax.debug.print('{}', mask.shape)

    r = lax.cond(
        mask,
        lambda : calc_r_from_s(s, s_offsets, g, n_e, n_i),
        lambda : jnp.zeros(s.shape),
    )

    v = W @ r + w_u * u(t)
    delta_s = (v - s) / tau_s
    delta_r_exp = (r[:, None] - r_exp) / tau_rules

    # Convenience functions
    def ds1(x, start, size):
        return lax.dynamic_slice(x, (start,), (size,))

    def ds2(x, start0, start1, size0, size1):
        return lax.dynamic_slice(x, (start0, start1), (size0, size1))

    # (1)->(1)
    delta_W_11_2f, delta_syn_11_2f_raw = delta_W_ij_two_factor(
        ds2(W, 0, 0, n_1, n_1) * W_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds2(r_exp, 0, 0, n_1, 12) * R_EXP_RESCALING,
        ds2(r_exp, 0, 0, n_1, 12) * R_EXP_RESCALING,
        ds1(c, 0, 20),
    )
    delta_syn_11_2f = delta_syn_11_2f_raw.sum()

    delta_W_11_3f, delta_syn_11_3f_raw = delta_W_ij_three_factor(
        ds2(W, 0, 0, n_1, n_1) * W_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds2(r_exp, 0, 36, n_1, 8) * R_EXP_RESCALING,
        ds2(r_exp, 0, 36, n_1, 8) * R_EXP_RESCALING,
        ds2(W, 0, n_1, n_1, n_2) @ ds2(r_exp, n_1, 44, n_2, 8) * R_EXP_RESCALING,
        ds1(c, 60, 8),
    )
    delta_syn_11_3f = delta_syn_11_3f_raw.sum()
    delta_W_11 = delta_W_11_2f + delta_W_11_3f

    # (2)->(1)
    delta_W_21_2f, delta_syn_21_2f_raw = delta_W_ij_two_factor(
        ds2(W, n_1, 0, n_2, n_1) * W_RESCALING,
        ds1(r, n_1, n_2) * R_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds2(r_exp, n_1, 12, n_2, 12) * R_EXP_RESCALING,
        ds2(r_exp, 0, 12, n_1, 12) * R_EXP_RESCALING,
        ds1(c, 20, 20),
    )
    delta_syn_21_2f = delta_syn_21_2f_raw.sum()

    # (1)->(2)
    delta_W_12_2f, delta_syn_12_2f_raw = delta_W_ij_two_factor(
        ds2(W, 0, n_1, n_1, n_2) * W_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds1(r, n_1, n_2) * R_RESCALING,
        ds2(r_exp, 0, 24, n_1, 12) * R_EXP_RESCALING,
        ds2(r_exp, n_1, 24, n_2, 12) * R_EXP_RESCALING,
        ds1(c, 40, 20),
    )
    delta_syn_12_2f = delta_syn_12_2f_raw.sum()

    delta_W_12_3f, delta_syn_12_3f_raw = delta_W_ij_three_factor(
        ds2(W, 0, n_1, n_1, n_2) * W_RESCALING,
        ds1(r, 0, n_1) * R_RESCALING,
        ds1(r, n_1, n_2) * R_RESCALING,
        ds2(r_exp, 0, 52, n_1, 8) * R_EXP_RESCALING,
        ds2(r_exp, n_1, 52, n_2, 8) * R_EXP_RESCALING,
        ds2(W, 0, 0, n_1, n_1) @ ds2(r_exp, 0, 60, n_1, 8) * R_EXP_RESCALING,
        ds1(c, 68, 8),
    )
    delta_syn_12_3f = delta_syn_12_3f_raw.sum()
    delta_W_12 = delta_W_12_2f + delta_W_12_3f

    delta_a = eta * w_nonzero * jnp.block([
        [delta_W_11, delta_W_12, jnp.zeros((n_1, n_i))],
        [delta_W_21_2f, jnp.zeros((n_2, n_2 + n_i))],
        [jnp.zeros((n_i, n_e + n_i))]
    ])
    delta_syn = eta * (
        delta_syn_11_2f +
        delta_syn_21_2f +
        delta_syn_12_2f +
        delta_syn_11_3f +
        delta_syn_12_3f
    )

    return delta_s, delta_r_exp, delta_a * mask, delta_syn * mask, delta_unstable & (~unstable_bool)


def simulate(
    t,
    a0,
    w_polarity,
    w_nonzero,
    r_in,
    c,
    tau_rules,
    n,
    dt,
    readout_times,
    args,
    save_for_viewing=False
):
    # Unpack static args
    time, g, s_offsets, w_u, tau_s, eta, n_e, n_i, n_e_pool, n_e_side = args
    num_devices = len(jax.devices())
    batch_shape = a0.shape[0]
    print(batch_shape, num_devices)
    assert batch_shape % num_devices == 0
    batch_per_device = batch_shape // num_devices

    # Reshape inputs to (num_devices, batch_per_device, ...)
    def reshape(x):
        return x.reshape((num_devices, batch_per_device) + x.shape[1:])

    a0 = reshape(a0)
    r_in = reshape(r_in)
    w_polarity = reshape(w_polarity)
    w_nonzero = reshape(w_nonzero)
    c = reshape(c)
    tau_rules = reshape(tau_rules)

    # Prepare readout/save times
    if save_for_viewing:
        viewing_points = jnp.linspace(t.min(), t.max(), 1000)
        merged_save_times, indices_viewing, indices_readout = merge_with_indices_jax(viewing_points, readout_times)
        save_times = merged_save_times
        sorted_indices = jnp.concatenate((indices_readout, indices_viewing))
    else:
        save_times = readout_times

    saveat = diffrax.SaveAt(ts=save_times)
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)

    # Define per-instance solver
    def solve_single_instance(a0, r_in, w_polarity, w_nonzero, c, tau_rules):
        s0 = jnp.zeros((n,))
        r_exp0 = jnp.zeros((n, tau_rules.shape[0]))
        syn0 = jnp.zeros(())
        unstable = jnp.zeros((), dtype=int)

        def learning_dynamics(t_, y, _):
            return _learning_dynamics(
                t_,
                y,
                r_in,
                w_polarity,
                w_nonzero,
                c,
                tau_rules,
                time,
                g,
                s_offsets,
                w_u,
                tau_s,
                eta,
                n_e,
                n_i,
                n_e_pool,
                n_e_side
            )

        term = diffrax.ODETerm(learning_dynamics)
        solver = diffrax.Tsit5()

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt,
            y0=(s0, r_exp0, a0, syn0, unstable),
            args=None,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=int(1e4),
        )
        return sol.ys  # tuple of arrays (time, n, ...)

    # Vectorize across batch_per_device (inner batch)
    vmapped_solver = jax.vmap(solve_single_instance, in_axes=0)

    # Parallelize across devices
    pmapped_solver = jax.pmap(vmapped_solver, axis_name="devices")

    # Run simulation
    results = pmapped_solver(a0, r_in, w_polarity, w_nonzero, c, tau_rules)
    return results
