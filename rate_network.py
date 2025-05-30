import numpy as np
from copy import deepcopy as copy
import jax
import jax.numpy as jnp
import diffrax
import jax.random as jr
from aux_funcs import merge_with_indices_jax

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


def calc_r_from_s(s, s_offsets, g, n_e):
    s_thresh = jnp.maximum(s - s_offsets, 0)
    r = g * jnp.concatenate((jnp.tanh(s_thresh[:n_e]), s_thresh[n_e:])) # excitatory cells get a tanh threshold, inhibition is left as threshold linear
    return r


def learning_dynamics(t, y, args):
    c, tau_rules, g, s_offsets, w_u, tau_s, eta, n_e, n_i, n_e_pool, n_e_side, time, r_in, w_polarity, w_nonzero = args

    def u(t_prime):
        return jax.vmap(jnp.interp, (None, None, 1),)(t_prime, time, r_in)

    n_1 = n_e_pool
    n_2 = 2 * n_e_side
    n_plastic = n_1 + n_2

    s, r_exp, a, syn, unstable = y
    W = w_polarity * softplus(a) * w_nonzero
    unstable_bool = unstable > 0

    pool_weights_zero = jnp.all(a[:n_e_pool, :n_e_pool] < 1e-6)
    pool_side_weights_zero = jnp.all(a[:n_e_pool, n_e_pool:n_e_pool + 2 * n_e_side] < 1e-6)
    side_pool_weights_zero = jnp.all(a[n_e_pool:n_e_pool + 2 * n_e_side, :n_e_pool] < 1e-6)
    weights_blew_up = jnp.any(a > 20)
    activity_blew_up = jnp.any(s > 10)

    delta_unstable = pool_weights_zero | pool_side_weights_zero | side_pool_weights_zero | weights_blew_up | activity_blew_up | unstable_bool

    r = calc_r_from_s(s, s_offsets, g, n_e) * ~(delta_unstable | unstable_bool)
    v = W @ r + w_u * u(t)
    delta_s = (v - s) / tau_s
    delta_r_exp = (r[:, None] - r_exp) / tau_rules

    # Weight change from (1) -> (1)

    delta_W_11_two_factor, delta_syn_11_two_factor_raw = delta_W_ij_two_factor(
        W[:n_1, :n_1] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, :12] * R_EXP_RESCALING,
        r_exp[:n_1, :12] * R_EXP_RESCALING,
        c[:20],
    )

    delta_syn_11_two_factor = delta_syn_11_two_factor_raw.sum()

    delta_W_11_three_factor, delta_syn_11_three_factor_raw = delta_W_ij_three_factor(
        W[:n_1, :n_1] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, 36:44] * R_EXP_RESCALING,
        r_exp[:n_1, 36:44] * R_EXP_RESCALING,
        # (2) -> (1)
        W[:n_1, n_1:n_plastic] @ r_exp[n_1:n_plastic, 44:52] * R_EXP_RESCALING,
        c[60:68],
    )

    delta_syn_11_three_factor = delta_syn_11_three_factor_raw.sum()

    delta_W_11 = (
        # (1) -> (1)
        delta_W_11_two_factor
        # (1) -> (1) modulated by (2)
        + delta_W_11_three_factor 
    )

    # Weight change from (2) -> (1)

    delta_W_21_two_factor, delta_syn_21_two_factor_raw = delta_W_ij_two_factor(
        W[n_1:n_plastic, :n_1] * W_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[n_1:n_plastic, 12:24] * R_EXP_RESCALING,
        r_exp[:n_1, 12:24] * R_EXP_RESCALING,
        c[20:40],
    )

    delta_syn_21_two_factor = delta_syn_21_two_factor_raw.sum()

    # Weight change from (1) -> (2)

    delta_W_12_two_factor, delta_syn_12_two_factor_raw  = delta_W_ij_two_factor(
        W[:n_1, n_1:n_plastic] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 24:36] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 24:36] * R_EXP_RESCALING,
        c[40:60],
    )

    delta_syn_12_two_factor =  delta_syn_12_two_factor_raw.sum()

    delta_W_12_three_factor, delta_syn_12_three_factor_raw = delta_W_ij_three_factor(
        W[:n_1, n_1:n_plastic] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 52:60] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 52:60] * R_EXP_RESCALING,
        # (1) -> (1)
        W[:n_1, :n_1] @ r_exp[:n_1, 60:68] * R_EXP_RESCALING,
        c[68:76],
    )

    delta_syn_12_three_factor = delta_syn_12_three_factor_raw.sum()

    delta_W_12 = (
        # (2) -> (1)
        delta_W_12_two_factor
        # (2) -> (1) modulated by (1)
        + delta_W_12_three_factor
    )

    delta_a = eta * w_nonzero * jnp.block(
        [
            [delta_W_11, delta_W_12, jnp.zeros((n_1, n_i),)],
            [delta_W_21_two_factor, jnp.zeros((n_2, n_2 + n_i))],
            [jnp.zeros((n_i, n_e + n_i))]
        ]
    )
    
    delta_syn = eta * (
        delta_syn_11_two_factor + 
        delta_syn_21_two_factor +
        delta_syn_12_two_factor +
        delta_syn_11_three_factor +
        delta_syn_12_three_factor
    )

    return delta_s, delta_r_exp, delta_a * ~(delta_unstable | unstable_bool), delta_syn * ~(delta_unstable | unstable_bool), delta_unstable & (~unstable_bool)


def simulate(t, a0, w_polarity, w_nonzero, r_in, c, tau_rules, n, dt, readout_times, args, save_for_viewing=False):
    s0 = jnp.zeros((a0.shape[0], n))
    r_exp0 = jnp.zeros((a0.shape[0], n, tau_rules.shape[1]))
    syn0 = jnp.zeros((a0.shape[0],))
    unstable = jnp.zeros((a0.shape[0],), dtype=int)

    term = diffrax.ODETerm(
        jax.vmap(
            learning_dynamics,
            (None, (0,) * 5, (0,) * 2 + (None,) * 10 + (0,) * 3),
        )
    )
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)

    if save_for_viewing:
        viewing_points = jnp.linspace(t.min(), t.max(), 1000)
        merged_save_times, indices_viewing, indices_readout = merge_with_indices_jax(viewing_points, readout_times)
        sorted_indices = jnp.concatenate((indices_readout, indices_viewing))
        saveat = diffrax.SaveAt(ts=merged_save_times)
    else:
        saveat = diffrax.SaveAt(ts=readout_times)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt,
        y0=(s0, r_exp0, a0, syn0, unstable),
        args=args + (t, r_in, w_polarity, w_nonzero),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e4),
    )

    finished_sol = jax.block_until_ready(sol)

    if save_for_viewing:
        return [y[sorted_indices, ...] for y in finished_sol.ys]
    else:
        return finished_sol.ys
