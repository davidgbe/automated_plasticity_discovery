import numpy as np
from copy import deepcopy as copy
import jax
import jax.numpy as jnp
import diffrax
import jax.random as jr

R_RESCALING = 5
R_EXP_RESCALING = 5

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
    return c * _delta_W_ij_two_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j)


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
        w_ij * r_exp_i[5] * w_ij * r_j * f_i[5],
        w_ij * r_exp_j[6] * f_i[6],
        w_ij * r_exp_j[7] * r_i * f_i[7],
    ])


@jax.jit
def _delta_W_ij_three_factor(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i, c):
    return c * _delta_W_ij_three_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i)


delta_W_ij_three_factor = jax.vmap(
    jax.vmap(_delta_W_ij_three_factor, (0, None, 0, None, 0, None, None)),
    (0, 0, None, 0, None, 0, None),
)


def calc_r_from_s(s, s_offsets, g, n_e):
    s_thresh = jnp.maximum(s - s_offsets, 0)
    r = g * jnp.concatenate((jnp.tanh(s_thresh[:n_e]), s_thresh[n_e:])) # excitatory cells get a tanh threshold, inhibition is left as threshold linear
    return r


def learning_dynamics(t, y, args):
    c, tau_rules, g, s_offsets, w_u, tau_s, eta, n_e, n_i, n_e_pool, n_e_side, time, r_in = args

    def u(t_prime):
        return jax.vmap(jnp.interp, (None, None, 1),)(t_prime, time, r_in)

    n_1 = n_e_pool
    n_2 = 2 * n_e_side
    n_plastic = n_1 + n_2

    s, r_exp, W, syn, unstable = y
    unstable_int = unstable.astype(int)

    delta_unstable = jnp.any(jnp.abs(W) > 20) | jnp.any(s > 10) | unstable_int

    r = calc_r_from_s(s, s_offsets, g, n_e) * ~(delta_unstable | unstable_int)
    v = W @ r + w_u * u(t)
    delta_s = (v - s) / tau_s
    delta_r_exp = (r[:, None] - r_exp) / tau_rules

    # Weight change from (1) -> (1)

    delta_W_11_two_factor_raw = delta_W_ij_two_factor(
        W[:n_1, :n_1],
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, :12] * R_EXP_RESCALING,
        r_exp[:n_1, :12] * R_EXP_RESCALING,
        c[:20],
    )

    delta_W_11_two_factor = delta_W_11_two_factor_raw.sum(axis=2)
    delta_syn_11_two_factor = jnp.abs(delta_W_11_two_factor_raw).sum(axis=(0, 1))
    del delta_W_11_two_factor_raw

    delta_W_11_three_factor_raw = delta_W_ij_three_factor(
        W[:n_1, :n_1],
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, 36:44] * R_EXP_RESCALING,
        r_exp[:n_1, 36:44] * R_EXP_RESCALING,
        # (2) -> (1)
        W[:n_1, n_1:n_plastic] @ r_exp[n_1:n_plastic, 44:52] * R_EXP_RESCALING,
        c[60:68],
    )

    delta_W_11_three_factor = delta_W_11_three_factor_raw.sum(axis=2)
    delta_syn_11_three_factor = jnp.abs(delta_W_11_three_factor_raw).sum(axis=(0, 1))
    del delta_W_11_three_factor_raw

    delta_W_11 = (
        # (1) -> (1)
        delta_W_11_two_factor
        # (1) -> (1) modulated by (2)
        + delta_W_11_three_factor 
    )

    # Weight change from (2) -> (1)

    delta_W_21_two_factor_raw = delta_W_ij_two_factor(
        W[n_1:n_plastic, :n_1],
        r[n_1:n_plastic] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[n_1:n_plastic, 12:24] * R_EXP_RESCALING,
        r_exp[:n_1, 12:24] * R_EXP_RESCALING,
        c[20:40],
    )

    delta_W_21_two_factor = delta_W_21_two_factor_raw.sum(axis=2)
    delta_syn_21_two_factor = jnp.abs(delta_W_21_two_factor_raw).sum(axis=(0, 1))
    del delta_W_21_two_factor_raw

    # Weight change from (1) -> (2)

    delta_W_12_two_factor_raw = delta_W_ij_two_factor(
        W[:n_1, n_1:n_plastic],
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 24:36] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 24:36] * R_EXP_RESCALING,
        c[40:60],
    )

    delta_W_12_two_factor = delta_W_12_two_factor_raw.sum(axis=2)
    delta_syn_12_two_factor = jnp.abs(delta_W_12_two_factor_raw).sum(axis=(0, 1))
    del delta_W_12_two_factor_raw

    delta_W_12_three_factor_raw = delta_W_ij_three_factor(
        W[:n_1, n_1:n_plastic],
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 52:60] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 52:60] * R_EXP_RESCALING,
        # (1) -> (1)
        W[:n_1, :n_1] @ r_exp[:n_1, 60:68] * R_EXP_RESCALING,
        c[68:76],
    )

    delta_W_12_three_factor = delta_W_12_three_factor_raw.sum(axis=2)
    delta_syn_12_three_factor = jnp.abs(delta_W_12_three_factor_raw).sum(axis=(0, 1))
    del delta_W_12_three_factor_raw

    delta_W_12 = (
        # (2) -> (1)
        delta_W_12_two_factor
        # (2) -> (1) modulated by (1)
        + delta_W_12_three_factor
    )

    delta_W = eta * jnp.block(
        [
            [delta_W_11, delta_W_12, jnp.zeros((n_1, n_i),)],
            [delta_W_21_two_factor, jnp.zeros((n_2, n_2 + n_i))],
            [jnp.zeros((n_i, n_e + n_i))]
        ]
    )

    delta_syn = eta * jnp.concatenate([
        delta_syn_11_two_factor,
        delta_syn_21_two_factor,
        delta_syn_12_two_factor,
        delta_syn_11_three_factor,
        delta_syn_12_three_factor,
    ])

    return delta_s, delta_r_exp, delta_W, delta_syn, delta_unstable & (~unstable_int)


def simulate(t, w, w_plastic, r_in, c, tau_rules, n, dt, readout_times, args, save_for_viewing=False):
    s0 = jnp.zeros((w.shape[0], n))
    r_exp0 = jnp.zeros((w.shape[0], n, tau_rules.shape[1]))
    W0 = w
    syn0 = jnp.zeros((w.shape[0], c.shape[1]))
    unstable = jnp.zeros((w.shape[0],), dtype=int)

    term = diffrax.ODETerm(
        jax.vmap(
            learning_dynamics,
            (None, (0,) * 5, (0,) * 2 + (None,) * 10 + (0,)),
        )
    )
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

    if save_for_viewing:
        saveat = diffrax.SaveAt(ts=jnp.linspace(t.min(), t.max(), 1000))
    else:
        saveat = diffrax.SaveAt(ts=readout_times)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt,
        y0=(s0, r_exp0, w, syn0, unstable),
        args=args + (t, r_in),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return jax.block_until_ready(sol)
