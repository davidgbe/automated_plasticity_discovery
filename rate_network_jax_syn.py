import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple, Dict, Any, Optional

R_RESCALING = 5
R_EXP_RESCALING = 5
W_RESCALING = 1
THREE_FACTOR_RESCALING = 2
SUMMED_WEIGHT_RESCALING = 0.05

# Layout of delta_syn_terms (100 values total):
#   [0:24]   pairwise terms, group 11
#   [24:48]  pairwise terms, group 21
#   [48:72]  pairwise terms, group 12
#   [72:76]  summed-weight terms, group 11
#   [76:80]  summed-weight terms, group 21
#   [80:84]  summed-weight terms, group 12
#   [84:92]  triplet terms, group 11
#   [92:100] triplet terms, group 12
N_PAIRWISE_TERMS = 24
N_SUMMED_WEIGHT_TERMS = 4
N_TRIPLET_TERMS = 8
N_SYN_TERMS = N_PAIRWISE_TERMS * 3 + N_SUMMED_WEIGHT_TERMS * 3 + N_TRIPLET_TERMS * 2  # 100

# PAIRWISE RULE LOGIC

@jax.jit
def _delta_W_ij_two_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j):
    return jnp.array(
        [
            jnp.float32(1),
            r_i,
            r_j,
            jnp.power(r_i, 2),
            jnp.power(r_j, 2),
            jnp.power(r_i, 3),
            jnp.power(r_j, 3),
            jnp.power(r_i, 4),
            jnp.power(r_j, 4),
            r_i * r_j,
            r_exp_i[0] * r_j,
            r_exp_j[1] * r_i,
            w_ij,
            w_ij * r_i,
            w_ij * r_j,
            w_ij * jnp.power(r_i, 2),
            w_ij * jnp.power(r_j, 2),
            w_ij * jnp.power(r_i, 3),
            w_ij * jnp.power(r_j, 3),
            w_ij * jnp.power(r_i, 4),
            w_ij * jnp.power(r_j, 4),
            w_ij * r_i * r_j,
            w_ij * r_exp_i[2] * r_j,
            w_ij * r_exp_j[3] * r_i,
        ]
    )

@jax.jit
def _delta_W_ij_two_factor(w_ij, r_i, r_j, r_exp_i, r_exp_j, c):
    # weighted per-term vector; shape (n_pairwise_terms,)
    delta_w_terms = c * _delta_W_ij_two_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j)
    return delta_w_terms.sum(), delta_w_terms

delta_W_ij_two_factor = jax.vmap(
    jax.vmap(_delta_W_ij_two_factor, (0, None, 0, None, 0, None)),
    (0, 0, None, 0, None, None),
)

# SUMMED WEIGHT BOUND RULE LOGIC

@partial(jax.jit, static_argnames=['W_shape_1', 'W_shape_2'])
def delta_W_ij_summed_weight_rules(W, c, W_shape_1, W_shape_2):
    delta_w_incoming = jnp.sum(W, axis=0)[None, :]
    delta_w_outgoing = jnp.sum(W, axis=1)[:, None]

    stacked_deltas = jnp.array([
        jnp.repeat(delta_w_incoming, repeats=W_shape_1, axis=0),
        jnp.repeat(delta_w_outgoing, repeats=W_shape_2, axis=1),
        delta_w_incoming * W,
        delta_w_outgoing * W,
    ])

    delta_w_per_term = c.reshape(c.shape[0], 1, 1) * stacked_deltas
    delta_w_terms = delta_w_per_term.sum(axis=(1, 2))

    return delta_w_per_term.sum(axis=0), delta_w_terms, delta_w_per_term

# THREE FACTOR RULE LOGIC

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
    ]) * THREE_FACTOR_RESCALING


@jax.jit
def _delta_W_ij_three_factor(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i, c):
    # weighted per-term vector; shape (n_triplet_terms,)
    delta_w_terms = c * _delta_W_ij_three_factor_rules(w_ij, r_i, r_j, r_exp_i, r_exp_j, f_i)
    return delta_w_terms.sum(), delta_w_terms


delta_W_ij_three_factor = jax.vmap(
    jax.vmap(_delta_W_ij_three_factor, (0, None, 0, None, 0, None, None)),
    (0, 0, None, 0, None, 0, None),
)

# Compute firing rates from synaptic activations

def calc_r_from_s(s, s_offsets, g, n_e):
    s_thresh = jnp.maximum(s - s_offsets, 0)
    r = g * jnp.concatenate((jnp.tanh(s_thresh[:n_e]), s_thresh[n_e:])) # excitatory cells get a tanh threshold, inhibition is left as threshold linear
    return r

# Helper function to enforce polarity of synapses and also keep size = 0 synapses at zero

def _enforce_polarity_and_structure(w, w_polarity, w0):
    return jax.lax.cond(
        (w * w_polarity > 0) & (w0 != 0),
        lambda w_: w_,
        lambda _: 0.0,
        w,
    )

enforce_polarity_and_structure = jax.vmap(
   jax.vmap(_enforce_polarity_and_structure, (0, 0, 0)),
   (0, 0, 0),
)

# Calculate changes in variables of interest for a single timestep

@partial(jax.jit, static_argnames=['n_e', 'n_i', 'n_e_pool', 'n_e_side', 'n_pairwise_rules', 'n_summed_weight_rules', 'n_triplet_rules'])
def learning_dynamics(
    s : jnp.ndarray,
    r_exp: jnp.ndarray,
    w : jnp.ndarray,
    syn : jnp.ndarray,
    unstable : bool,
    u : jnp.ndarray,
    args : Tuple,
    n_e : int,
    n_i : int,
    n_e_pool : int,
    n_e_side : int,
    n_pairwise_rules : int,
    n_summed_weight_rules : int,
    n_triplet_rules : int,
):
    coef_offset = n_pairwise_rules + n_summed_weight_rules
    triplet_rule_start = 3 * coef_offset

    c, tau_rules, g, s_offsets, w_u, tau_s, eta, dt = args

    n_1 = n_e_pool
    n_2 = 2 * n_e_side
    n_plastic = n_1 + n_2

    abs_w = jnp.abs(w)

    pool_weights_zero = jnp.all(abs_w[:n_e_pool, :n_e_pool] < 1e-6)
    pool_side_weights_zero = jnp.all(abs_w[:n_e_pool, n_e_pool:n_e_pool + 2 * n_e_side] < 1e-6)
    side_pool_weights_zero = False
    weights_blew_up = jnp.any(abs_w > 20)
    activity_blew_up = jnp.any(s > 20)

    unstable = pool_weights_zero | pool_side_weights_zero | side_pool_weights_zero | weights_blew_up | activity_blew_up | unstable

    r = calc_r_from_s(s, s_offsets, g, n_e)
    v = w @ r + w_u * u
    delta_s = (v - s) * dt / tau_s
    delta_r_exp = (r[:, None] - r_exp) * dt / tau_rules

    # Masks based on current w
    mask_11 = abs_w[:n_1, :n_1] > 1e-6                    # (n_1, n_1)
    mask_21 = abs_w[n_1:n_plastic, :n_1] > 1e-6           # (n_2, n_1)
    mask_12 = abs_w[:n_1, n_1:n_plastic] > 1e-6           # (n_1, n_2)

    # Weight change from (1) -> (1)

    delta_W_11_two_factor, delta_terms_11_pairwise = delta_W_ij_two_factor(
        w[:n_1, :n_1] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, :4] * R_EXP_RESCALING,
        r_exp[:n_1, :4] * R_EXP_RESCALING,
        c[:n_pairwise_rules],
    )
    delta_terms_11_pairwise = (delta_terms_11_pairwise * mask_11[..., None]).sum(axis=(0, 1))

    delta_W_11_summed_weight, _, delta_w_per_term_11 = delta_W_ij_summed_weight_rules(
        w[:n_1, :n_1] * SUMMED_WEIGHT_RESCALING,
        c[n_pairwise_rules:n_pairwise_rules + n_summed_weight_rules],
        W_shape_1=n_1,
        W_shape_2=n_1,
    )
    delta_terms_11_summed = (delta_w_per_term_11 * mask_11[None, ...]).sum(axis=(1, 2))

    delta_W_11_three_factor, delta_terms_11_triplet = delta_W_ij_three_factor(
        w[:n_1, :n_1] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[:n_1, 12:20] * R_EXP_RESCALING,
        r_exp[:n_1, 12:20] * R_EXP_RESCALING,
        w[:n_1, n_1:n_plastic] @ r_exp[n_1:n_plastic, 20:28] * R_EXP_RESCALING * THREE_FACTOR_RESCALING,
        c[triplet_rule_start:triplet_rule_start + n_triplet_rules],
    )
    delta_terms_11_triplet = (delta_terms_11_triplet * mask_11[..., None]).sum(axis=(0, 1))

    delta_W_11 = (
        delta_W_11_two_factor
        + delta_W_11_summed_weight
        + delta_W_11_three_factor
    )

    # Weight change from (2) -> (1)

    delta_W_21_two_factor, delta_terms_21_pairwise = delta_W_ij_two_factor(
        w[n_1:n_plastic, :n_1] * W_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r[:n_1] * R_RESCALING,
        r_exp[n_1:n_plastic, 4:8] * R_EXP_RESCALING,
        r_exp[:n_1, 4:8] * R_EXP_RESCALING,
        c[coef_offset:coef_offset + n_pairwise_rules],
    )
    delta_terms_21_pairwise = (delta_terms_21_pairwise * mask_21[..., None]).sum(axis=(0, 1))

    delta_W_21_summed_weight, _, delta_w_per_term_21 = delta_W_ij_summed_weight_rules(
        w[n_1:n_plastic, :n_1] * SUMMED_WEIGHT_RESCALING,
        c[coef_offset + n_pairwise_rules:coef_offset + n_pairwise_rules + n_summed_weight_rules],
        W_shape_1=n_2,
        W_shape_2=n_1,
    )
    delta_terms_21_summed = (delta_w_per_term_21 * mask_21[None, ...]).sum(axis=(1, 2))

    delta_W_21 = (
        delta_W_21_two_factor
        + delta_W_21_summed_weight
    )

    # Weight change from (1) -> (2)

    delta_W_12_two_factor, delta_terms_12_pairwise = delta_W_ij_two_factor(
        w[:n_1, n_1:n_plastic] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 8:12] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 8:12] * R_EXP_RESCALING,
        c[2 * coef_offset : 2 * coef_offset + n_pairwise_rules],
    )
    delta_terms_12_pairwise = (delta_terms_12_pairwise * mask_12[..., None]).sum(axis=(0, 1))

    delta_W_12_summed_weight, _, delta_w_per_term_12 = delta_W_ij_summed_weight_rules(
        w[:n_1, n_1:n_plastic] * SUMMED_WEIGHT_RESCALING,
        c[2 * coef_offset + n_pairwise_rules:2 * coef_offset + n_pairwise_rules + n_summed_weight_rules],
        W_shape_1=n_1,
        W_shape_2=n_2,
    )
    delta_terms_12_summed = (delta_w_per_term_12 * mask_12[None, ...]).sum(axis=(1, 2))

    delta_W_12_three_factor, delta_terms_12_triplet = delta_W_ij_three_factor(
        w[:n_1, n_1:n_plastic] * W_RESCALING,
        r[:n_1] * R_RESCALING,
        r[n_1:n_plastic] * R_RESCALING,
        r_exp[:n_1, 28:36] * R_EXP_RESCALING,
        r_exp[n_1:n_plastic, 28:36] * R_EXP_RESCALING,
        w[:n_1, :n_1] @ r_exp[:n_1, 36:44] * R_EXP_RESCALING * THREE_FACTOR_RESCALING,
        c[triplet_rule_start + n_triplet_rules:triplet_rule_start + 2 * n_triplet_rules],
    )
    delta_terms_12_triplet = (delta_terms_12_triplet * mask_12[..., None]).sum(axis=(0, 1))

    delta_W_12 = (
        delta_W_12_two_factor
        + delta_W_12_summed_weight
        + delta_W_12_three_factor
    )

    # Assemble delta_syn_terms
    delta_syn_factors = eta * dt * jnp.concatenate([
        delta_terms_11_pairwise,
        delta_terms_21_pairwise,
        delta_terms_12_pairwise,
        delta_terms_11_summed,
        delta_terms_21_summed,
        delta_terms_12_summed,
        delta_terms_11_triplet,
        delta_terms_12_triplet,
    ])

    row1 = jnp.concatenate([
        delta_W_11,
        delta_W_12,
        jnp.zeros((n_1, n_i))
    ], axis=1)

    row2 = jnp.concatenate([
        delta_W_21,
        jnp.zeros((n_2, n_2 + n_i))
    ], axis=1)

    row3 = jnp.zeros((n_i, n_e + n_i))

    delta_w = eta * dt * jnp.concatenate([
        row1,
        row2,
        row3,
    ], axis=0)

    delta_syn = delta_syn_factors.sum()

    return r, delta_s, delta_r_exp, delta_w, delta_syn, delta_syn_factors, unstable

# Simulate a full unroll of network dynamics for len_t timesteps

@partial(jax.jit, static_argnames=['len_t', 'n_timeconsts', 'n_e', 'n_i', 'n_e_pool', 'n_e_side', 'n_rules', 'n_pairwise_rules', 'n_summed_weight_rules', 'n_triplet_rules'])
def simulate_save_syn(
    len_t,
    n_timeconsts,
    dt,
    t,
    w0,
    r_in,
    c,
    tau_rules,
    g,
    s_offsets : jnp.ndarray,
    w_u : float,
    tau_s : float,
    eta : float,
    n_e : int,
    n_i : int,
    n_e_pool : int,
    n_e_side : int,
    n_rules : int,
    n_pairwise_rules : int,
    n_summed_weight_rules : int,
    n_triplet_rules : int,
):

    args = (c, tau_rules, g, s_offsets, w_u, tau_s, eta, dt)

    s0 = jnp.zeros((n_e + n_i))
    r_exp0 = jnp.zeros((n_e + n_i, n_timeconsts))
    syn0 = jnp.zeros((n_rules,))
    syn_factors0 = jnp.zeros((n_rules,))
    unstable0 = False

    w_polarity = (-1 + 2 * (w0 >= 0).astype(int)).astype(int)

    def scan(carry, r_in):
        s, r_exp, w, syn, syn_factors, unstable = carry

        r, ds, dr_exp, dw, dsyn, dsyn_factors, unstable = learning_dynamics(
            s=s,
            r_exp=r_exp,
            w=w,
            syn=syn,
            unstable=unstable,
            u=r_in,
            args=args,
            n_e=n_e,
            n_i=n_i,
            n_e_pool=n_e_pool,
            n_e_side=n_e_side,
            n_pairwise_rules=n_pairwise_rules,
            n_summed_weight_rules=n_summed_weight_rules,
            n_triplet_rules=n_triplet_rules,
        )

        s_prime, r_exp_prime, w_prime, syn_prime, syn_factors_prime = jax.lax.cond(
            unstable,
            lambda _: (
                jnp.zeros_like(s),
                jnp.zeros_like(r_exp),
                jnp.zeros_like(w),
                jnp.zeros_like(syn),
                jnp.zeros_like(syn_factors),
            ),
            lambda _: (
                s + ds,
                r_exp + dr_exp,
                enforce_polarity_and_structure(w + dw, w_polarity, w0),
                syn + dsyn,
                syn_factors + dsyn_factors,
            ),
            None,
        )

        return (s_prime, r_exp_prime, w_prime, syn_prime, syn_factors_prime, unstable), r


    (s, r_exp, w, syn, syn_factors, unstable), r = jax.lax.scan(
        scan,
        (s0, r_exp0, w0, syn0, syn_factors0, unstable0),
        r_in,
    )

    return r, w, syn, syn_factors, r_exp