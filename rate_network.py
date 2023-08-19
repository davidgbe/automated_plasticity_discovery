import numpy as np
from copy import deepcopy as copy
from numba import njit

### For initiating activity

def generate_gaussian_pulse(t, u, s, w=1):
    return w * np.exp(-0.5 * np.square((t-u) / s))

### Related to activation functions

def shift(x : np.ndarray):
    shifted = np.concatenate([[0], copy(x[:-1])])
    return shifted

def threshold_linear(s : np.ndarray, v_th : float):
    shifted_s = s - v_th
    shifted_s[shifted_s < 0] = 0
    return shifted_s

def tanh(s : np.ndarray, v_th : float):
    return np.tanh(threshold_linear(s, v_th))

def sigmoid(s : np.ndarray, v_th : float, spread : float):
    return 2 / (1 + np.exp(-1*(s - v_th) / spread))

def threshold_power(s : np.ndarray, v_th : float, p : float):
    return np.power(threshold_linear(s, v_th), p)

### Simulate dynamics
def simulate(t : np.ndarray, n_e : int, n_i : int, inp : np.ndarray, plasticity_coefs : np.ndarray, rule_time_constants : np.ndarray, w : np.ndarray, w_plastic : np.ndarray, tau_e=5e-3, tau_i=5e-3, dt=1e-6, g=1, w_u=1, track_params=False):    
    len_t = len(t)

    inh_activity = np.zeros((len_t))
    r = np.zeros((len_t, n_e + n_i))
    s = np.zeros((len_t, n_e + n_i))
    v = np.zeros((len_t, n_e + n_i))
    r_exp_filtered = np.zeros((len(rule_time_constants), len_t, n_e + n_i))

    sign_w = np.where(w >= 0, 1, -1).astype(int)
    inf_w = np.where(sign_w >= 0, 1e-6, -1e-6)

    tau = np.concatenate([tau_e * np.ones(n_e), tau_i * np.ones(n_i)])

    n_params = len(plasticity_coefs)

    w_copy, effects_e_e, effects_e_i, effects_i_e = simulate_inner_loop(t, n_e, n_i, inp, plasticity_coefs, rule_time_constants, w, w_plastic, dt, g, w_u, track_params, len_t, inh_activity, r, s, v, r_exp_filtered, sign_w, inf_w, tau, n_params)

    if track_params:
        return r, s, v, w_copy, np.concatenate([effects_e_e, effects_e_i, effects_i_e]), r_exp_filtered
    else:
        return r, s, v, w_copy, None, r_exp_filtered

@njit
def repeat_vec(vec : np.ndarray, n : int): # same as ones_vec or np.outer(vec, ones)
    outer = np.empty((vec.shape[0], n), dtype=np.float)
    for index in range(n):
        outer[:, index] = vec
    return outer

@njit
def outer_bin_vec_vec(bin_vec : np.ndarray, vec : np.ndarray):
    outer = np.empty((vec.shape[0], bin_vec.shape[0]), dtype=np.float)
    for index in range(outer.shape[1]):
        if bin_vec[index] > 0:
            outer[:, index] = vec
        else:
            outer[:, index] = 0
    return outer

@njit
def simulate_inner_loop(
    t : np.ndarray,
    n_e : int,
    n_i : int,
    inp : np.ndarray,
    plasticity_coefs : np.ndarray,
    rule_time_constants : np.ndarray,
    w : np.ndarray,
    w_plastic : np.ndarray,
    dt : float,
    g : float,
    w_u : float,
    track_params : bool,
    len_t : int,
    inh_activity : np.ndarray,
    r : np.ndarray,
    s : np.ndarray,
    v : np.ndarray,
    r_exp_filtered : np.ndarray,
    sign_w : np.ndarray,
    inf_w : np.ndarray,
    tau : np.ndarray,
    n_params : int):

    one_third_n_params = int(n_params)
    pop_slices = [slice(0, n_e), slice(n_e, n_e + n_i)]
    pop_sizes = [n_e, n_i]

    w_copy = np.copy(w)
    effects_e_e = np.zeros((one_third_n_params))
    effects_e_i = np.zeros((one_third_n_params))
    effects_i_e = np.zeros((one_third_n_params))

    int_time_consts = rule_time_constants.reshape(rule_time_constants.shape[0], 1) * np.ones((rule_time_constants.shape[0], n_e + n_i))

    n_total = n_e + n_i

    for i in range(0, len(t) - 1):
        v[i+1, :] = w_u * inp[i, :] + np.dot(w_copy, r[i, :].T) # calculate input to synaptic conductance equation
        s[i+1, :] = s[i, :] + (v[i+1, :] - s[i, :]) * dt / tau # update synaptic conductance as exponential filter of input

        # firing rates are calculated as normalized synaptic conductances
        shifted_s_e = s[i, :n_e] - 0.1
        shifted_s_e[shifted_s_e < 0] = 0
        r[i+1, :n_e] = g * np.tanh(shifted_s_e)

        shifted_s_i = s[i, n_e:] - 0.1
        shifted_s_i[shifted_s_i < 0] = 0
        r[i+1, n_e:] = g * np.tanh(shifted_s_i)
        
        # calculate exponential filtered of firing rate to use for STDP-like plasticity rules
        r_exp_filtered[:, i+1, :] = r_exp_filtered[:, i, :] * (1 - dt / int_time_consts) + r[i, :] * (dt / int_time_consts)

        r_0_pow = np.ones(n_e + n_i) / 5
        r_1_pow = r[i+1, :] / 0.01
        r_bin = np.zeros(n_e + n_i)
        r_bin[r_1_pow > 0] = 1
        # r_2_pow = np.square(r[i+1, :]) / 0.01
        r_exp_filtered_curr = r_exp_filtered[:, i+1, :] / 0.01

        r_0_pow_split = [r_0_pow[:n_e], r_0_pow[n_e:n_e + n_i]]
        r_bin_split = [r_bin[:n_e], r_bin[n_e:n_e + n_i]]
        r_1_pow_split = [r_1_pow[:n_e], r_1_pow[n_e:n_e + n_i]]
        r_exp_filtered_curr_split = [r_exp_filtered_curr[:, :n_e], r_exp_filtered_curr[:, n_e:n_e + n_i]]

        # find outer products of zeroth, first powers of firing rates to compute updates due to plasticity rules
        r_0_r_0 = np.outer(r_0_pow, r_0_pow)
  
        r_bin_r_0 = np.outer(r_0_pow, r_bin)
        r_0_r_bin = r_bin_r_0.T

        r_0_r_1 = np.outer(r_1_pow, r_0_pow)
        r_1_r_0 = r_0_r_1.T

        r_bin_r_bin = np.outer(r_bin, r_bin)
        r_bin_r_1 = np.outer(r_1_pow, r_bin)

        r_1_r_bin = r_bin_r_1.T
        r_1_r_1 = np.outer(r_1_pow, r_1_pow)

        w_updates_unweighted = []

        for k, pop_indices in enumerate([[0, 0]]): # [[0, 0], [0, 1], [1, 0]]
            p_i = pop_indices[0]
            p_j = pop_indices[1]


            r_1_r_exp = np.outer(r_exp_filtered_curr_split[p_j][k, :], r_1_pow_split[p_i])

            r_exp_r_1 = np.outer(r_1_pow_split[p_j], r_exp_filtered_curr_split[p_i][k + 1, :])

            # r_bin_r_exp = np.outer(r_exp_filtered_curr_split[p_j][k + 2, :], r_bin_split[p_i]) # same as np.outer(r_exp[p_j], r_bin[p_i])
            # r_exp_r_bin = np.outer(r_bin_split[p_j], r_exp_filtered_curr_split[p_i][k + 3, :])

            r_exp_r = r_exp_filtered_curr_split[p_j][k + 2, :] * r_1_pow_split[p_j]
            r_0_by_r_exp_r = np.outer(r_exp_r, r_0_pow_split[p_i])

            r_exp_r = r_exp_filtered_curr_split[p_i][k + 3, :] * r_1_pow_split[p_i]
            r_exp_r_by_r_0 = np.outer(r_0_pow_split[p_j], r_exp_r)

            r_exp_square = np.square(r_exp_filtered_curr_split[p_j][k + 4, :])
            r_0_by_r_exp_2 = np.outer(r_exp_square, r_0_pow_split[p_i])

            r_exp_square = np.square(r_exp_filtered_curr_split[p_i][k + 5, :])
            r_exp_2_by_r_0 = np.outer(r_0_pow_split[p_j], r_exp_square)


            r_1_r_exp_w = np.outer(r_exp_filtered_curr_split[p_j][k + 6, :], r_1_pow_split[p_i])

            r_exp_r_1_w = np.outer(r_1_pow_split[p_j], r_exp_filtered_curr_split[p_i][k + 7, :])

            # r_bin_r_exp_w = np.outer(r_exp_filtered_curr_split[p_j][k + 10, :], r_bin_split[p_i]) # same as np.outer(r_exp[p_j], r_bin[p_i])
            # r_exp_r_bin_w = np.outer(r_bin_split[p_j], r_exp_filtered_curr_split[p_i][k + 11, :])

            r_exp_r = r_exp_filtered_curr_split[p_j][k + 8, :] * r_1_pow_split[p_j]
            r_0_by_r_exp_r_w = np.outer(r_exp_r, r_0_pow_split[p_i])

            r_exp_r = r_exp_filtered_curr_split[p_i][k + 9, :] * r_1_pow_split[p_i]
            r_exp_r_by_r_0_w = np.outer(r_0_pow_split[p_j], r_exp_r)

            r_exp_square = np.square(r_exp_filtered_curr_split[p_j][k + 10, :])
            r_0_by_r_exp_2_w = np.outer(r_exp_square, r_0_pow_split[p_i])

            r_exp_square = np.square(r_exp_filtered_curr_split[p_i][k + 11, :])
            r_exp_2_by_r_0_w = np.outer(r_0_pow_split[p_j], r_exp_square)

            # print(r_1_r_exp.shape)
            # print(r_exp_r_1.shape)
            # print(r_bin_r_exp.shape)
            # print(r_exp_r_bin.shape)
            # print(r_0_by_r_exp_r.shape)
            # print(r_exp_r_by_r_0.shape)
            # print(r_0_by_r_exp_2.shape)
            # print(r_exp_2_by_r_0.shape)


            r_cross_products = np.stack((
                r_0_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_1[pop_slices[p_j], pop_slices[p_i]],

                # r_0_r_bin[pop_slices[p_j], pop_slices[p_i]],
                # r_bin_r_0[pop_slices[p_j], pop_slices[p_i]],
                # r_1_r_bin[pop_slices[p_j], pop_slices[p_i]],
                # r_bin_r_1[pop_slices[p_j], pop_slices[p_i]],
                # r_bin_r_bin[pop_slices[p_j], pop_slices[p_i]],

                r_1_r_exp,
                r_exp_r_1,
                # r_bin_r_exp,
                # r_exp_r_bin,
                r_0_by_r_exp_r,
                r_exp_r_by_r_0,
                r_0_by_r_exp_2,
                r_exp_2_by_r_0,


                r_0_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_1[pop_slices[p_j], pop_slices[p_i]],

                # r_bin_r_0[pop_slices[p_j], pop_slices[p_i]],
                # r_0_r_bin[pop_slices[p_j], pop_slices[p_i]],
                # r_1_r_bin[pop_slices[p_j], pop_slices[p_i]],
                # r_bin_r_1[pop_slices[p_j], pop_slices[p_i]],
                # r_bin_r_bin[pop_slices[p_j], pop_slices[p_i]],

                r_1_r_exp_w,
                r_exp_r_1_w,
                # r_bin_r_exp_w,
                # r_exp_r_bin_w,
                r_0_by_r_exp_r_w,
                r_exp_r_by_r_0_w,
                r_0_by_r_exp_2_w,
                r_exp_2_by_r_0_w,
            ))

            w_updates_unweighted.append(r_cross_products)
            num_rules = w_updates_unweighted[k].shape[0]
            w_updates_unweighted[k][int(0.5 * num_rules):num_rules] = w_copy[pop_slices[p_j], pop_slices[p_i]] * w_updates_unweighted[k][int(0.5 * num_rules):num_rules]

        w_not_almost_zero = np.where(np.abs(w) > 2e-6, 1, 0)

        if track_params:
            dw_e_e_unsummed = plasticity_coefs[:one_third_n_params].reshape(one_third_n_params, 1, 1) * (w_updates_unweighted[0] * w_plastic[:n_e, :n_e] * w_not_almost_zero[:n_e, :n_e])
            effects_e_e_delta = np.sum(np.abs(dw_e_e_unsummed), axis=1)
            effects_e_e_delta = np.sum(effects_e_e_delta, axis=1)
            effects_e_e += effects_e_e_delta

            # dw_e_i_unsummed = plasticity_coefs[one_third_n_params:2*one_third_n_params].reshape(one_third_n_params, 1, 1) * (w_updates_unweighted[1] * w_plastic[n_e:n_e + n_i, :n_e] * w_not_almost_zero[n_e:n_e + n_i, :n_e])
            # effects_e_i_delta = np.sum(np.abs(dw_e_i_unsummed), axis=1)
            # effects_e_i_delta = np.sum(effects_e_i_delta, axis=1)
            # effects_e_i += effects_e_i_delta

            # dw_i_e_unsummed = plasticity_coefs[2 * one_third_n_params:].reshape(one_third_n_params, 1, 1) * (w_updates_unweighted[2] * w_plastic[:n_e, n_e:n_e + n_i] * w_not_almost_zero[:n_e, n_e:n_e + n_i])
            # effects_i_e_delta = np.sum(np.abs(dw_i_e_unsummed), axis=1)
            # effects_i_e_delta = np.sum(effects_i_e_delta, axis=1)
            # effects_i_e += effects_i_e_delta

            dw_e_e = np.sum(dw_e_e_unsummed, axis=0)
            # dw_e_i = np.sum(dw_e_i_unsummed, axis=0)
            # dw_i_e = np.sum(dw_i_e_unsummed, axis=0)
        else:
            # dot updates due to all rules with coefficients for these rules and compute total weight updates. Do not update non-plastic weights.
            dw_e_e = np.sum(plasticity_coefs[:one_third_n_params].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[0], axis=0) * w_plastic[:n_e, :n_e]
            # dw_e_i = np.sum(plasticity_coefs[one_third_n_params:2*one_third_n_params].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[1], axis=0) * w_plastic[n_e:n_e + n_i, :n_e]
            # dw_i_e = np.sum(plasticity_coefs[2 * one_third_n_params:].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[2], axis=0) * w_plastic[:n_e, n_e:n_e + n_i]

        w_copy[:n_e, :n_e] += (0.0005 * dw_e_e)
        # w_copy[n_e:n_e + n_i, :n_e] += (0.0005 * dw_e_i)
        # w_copy[:n_e, n_e:n_e + n_i] += (0.0005 * dw_i_e)

        # if sign of weight is flipped by update, set it to an infinitesimal amount with its initial polarity
        polarity_flip = sign_w * w_copy
        w_copy = np.where(polarity_flip >= 0, w_copy, inf_w)

    return w_copy, effects_e_e, effects_e_i, effects_i_e

