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
def simulate(t : np.ndarray, n_e_pool : int, n_e_side : int, n_i : int, inp : np.ndarray, plasticity_coefs : np.ndarray, rule_time_constants : np.ndarray, w : np.ndarray, w_plastic : np.ndarray, tau_e=5e-3, tau_i=5e-3, dt=1e-6, g=1, w_u=1, track_params=False):    
    len_t = len(t)

    network_size = n_e_pool + 2 * n_e_side + n_i
    r = np.zeros((len_t, network_size))
    s = np.zeros((len_t, network_size))
    v = np.zeros((len_t, network_size))
    r_exp_filtered = np.zeros((len(rule_time_constants), len_t, network_size))

    sign_w = np.where(w >= 0, 1, -1).astype(int)
    inf_w = np.where(sign_w >= 0, 1e-6, -1e-6)

    tau = np.concatenate([tau_e * np.ones(n_e_pool + 2 * n_e_side), tau_i * np.ones(n_i)])

    n_params = len(plasticity_coefs)

    w_copy, effects = simulate_inner_loop(t, n_e_pool, n_e_side, n_i, inp, plasticity_coefs, rule_time_constants, w, w_plastic, dt, g, w_u, track_params, r, s, v, r_exp_filtered, sign_w, inf_w, tau, n_params)

    return r, s, v, w_copy, effects, r_exp_filtered

@njit
def simulate_inner_loop(
    t : np.ndarray,
    n_e_pool : int,
    n_e_side : int,
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
    r : np.ndarray,
    s : np.ndarray,
    v : np.ndarray,
    r_exp_filtered : np.ndarray,
    sign_w : np.ndarray,
    inf_w : np.ndarray,
    tau : np.ndarray,
    n_params : int):

    n_e = n_e_pool + 2 * n_e_side

    coefficient_for_division_pair = [slice(0, 20), slice(20, 40), slice(40, 60)]
    coefficient_for_division_trip = [slice(60, 64), slice(64, 68)]
    pop_slices = [slice(0, n_e_pool), slice(n_e_pool, n_e_pool + 2 * n_e_side)]

    w_copy = np.copy(w)
    effects = np.zeros((n_params))

    int_time_consts = rule_time_constants.reshape(rule_time_constants.shape[0], 1) * np.ones((rule_time_constants.shape[0], n_e + n_i))

    for i in range(0, len(t) - 1):
        v[i+1, :] = w_u * inp[i, :] + np.dot(w_copy, r[i, :].T) # calculate input to synaptic conductance equation
        s[i+1, :] = s[i, :] + (v[i+1, :] - s[i, :]) * dt / tau # update synaptic conductance as exponential filter of input

        # firing rates are calculated as normalized synaptic conductances
        shifted_s_e = s[i, :n_e] - 0.1
        shifted_s_e[shifted_s_e < 0] = 0
        r[i+1, :n_e] = g * np.tanh(shifted_s_e)

        shifted_s_i = s[i, n_e:]# - 0.1
        shifted_s_i[shifted_s_i < 0] = 0
        r[i+1, n_e:] = g * shifted_s_i
        
        # calculate exponential filtered of firing rate to use for STDP-like plasticity rules
        r_exp_filtered[:, i+1, :] = r_exp_filtered[:, i, :] * (1 - dt / int_time_consts) + r[i, :] * (dt / int_time_consts)

        r_0_pow = np.ones(n_e + n_i)
        r_1_pow = r[i+1, :] / 0.02
        r_2_pow = np.square(r[i+1, :]) / 0.01
        r_exp_filtered_curr = r_exp_filtered[:, i+1, :] / 0.01

        r_0_pow_split = [r_0_pow[s] for s in pop_slices]
        r_1_pow_split = [r_1_pow[s] for s in pop_slices]
        r_exp_filtered_curr_split = [r_exp_filtered_curr[:, s] for s in pop_slices]

        # find outer products of zeroth, first powers of firing rates to compute updates due to plasticity rules
        r_0_r_0 = np.outer(r_0_pow, r_0_pow)
        r_0_r_1 = np.outer(r_1_pow, r_0_pow)
        r_1_r_0 = r_0_r_1.T
        r_1_r_1 = np.outer(r_1_pow, r_1_pow)

        w_updates_unweighted = []

        for k, pop_indices in enumerate([[0, 0], [0, 1], [1, 0]]):
            p_i = pop_indices[0]
            p_j = pop_indices[1]
            ts_for_pop_start = k * 12

            r_0_r_exp = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start, :], r_0_pow_split[p_i])
            r_1_r_exp = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 1, :], r_1_pow_split[p_i])
            r_exp_r_0 = np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 2, :])
            r_exp_r_1 = np.outer(r_1_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 3, :])
            r_0_by_r_exp_r = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 4, :] * r_1_pow_split[p_j], r_0_pow_split[p_i])
            r_exp_r_by_r_0 = np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 5, :] * r_1_pow_split[p_i])

            r_0_r_exp_w = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 6, :], r_0_pow_split[p_i])
            r_1_r_exp_w = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 7, :], r_1_pow_split[p_i])
            r_exp_r_0_w = np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 8, :])
            r_exp_r_1_w = np.outer(r_1_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 9, :])
            r_0_by_r_exp_r_w = np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 10, :] * r_1_pow_split[p_j], r_0_pow_split[p_i])
            r_exp_r_by_r_0_w = np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 11, :] * r_1_pow_split[p_i])

            r_cross_products = np.stack((
                r_0_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_exp,
                r_1_r_exp,
                r_exp_r_0,
                r_exp_r_1,
                r_0_by_r_exp_r,
                r_exp_r_by_r_0,

                r_0_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_0[pop_slices[p_j], pop_slices[p_i]],
                r_1_r_1[pop_slices[p_j], pop_slices[p_i]],
                r_0_r_exp_w,
                r_1_r_exp_w,
                r_exp_r_0_w,
                r_exp_r_1_w,
                r_0_by_r_exp_r_w,
                r_exp_r_by_r_0_w,

            ))

            w_updates_unweighted.append(r_cross_products)
            num_rules = w_updates_unweighted[-1].shape[0]
            w_updates_unweighted[k][int(0.5 * num_rules):num_rules] = w_copy[pop_slices[p_j], pop_slices[p_i]] * w_updates_unweighted[k][int(0.5 * num_rules):num_rules]
        
        w_not_almost_zero = np.where(np.abs(w) > 2e-6, 1, 0)

        # compute the same for 3 factor rules
        # what's the operative synapse here?
        for k, pop_indices in enumerate([[0, 0, 1], [1, 0, 0]]):
            p_i = pop_indices[0]
            p_j = pop_indices[1]
            p_l = pop_indices[2]

            ts_for_pop_start = k * 8 + 36

            # how to read this:
            # first factor is presynaptic neuron from first population (p_i)
            # second factor is postsynaptic neuron from second population (p_j)
            # third factor is summmed integrated inputs from the thid population (p_l) to second pop (p_j) NOTE: this info is only local if `l`` is connected to `j`
            w_not_almost_zero_k = w_not_almost_zero[pop_slices[p_j], pop_slices[p_l]]
            
            third_factor = np.dot(w_not_almost_zero_k, r_exp_filtered_curr_split[p_l][ts_for_pop_start + 1, :])
            r_exp_r_1_r_exp_sum = third_factor.reshape(third_factor.shape[0], 1) * np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start, :], r_0_pow_split[p_i])
            third_factor = np.dot(w_not_almost_zero_k, r_exp_filtered_curr_split[p_l][ts_for_pop_start + 3, :])
            r_1_r_exp_r_exp_sum = third_factor.reshape(third_factor.shape[0], 1) * np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 2, :])

            third_factor = np.dot(w_not_almost_zero_k, r_exp_filtered_curr_split[p_l][ts_for_pop_start + 5, :])
            r_exp_r_1_r_exp_sum_w = third_factor.reshape(third_factor.shape[0], 1) * np.outer(r_exp_filtered_curr_split[p_j][ts_for_pop_start + 4, :], r_0_pow_split[p_i])
            third_factor = np.dot(w_not_almost_zero_k, r_exp_filtered_curr_split[p_l][ts_for_pop_start + 7, :])
            r_1_r_exp_r_exp_sum_w = third_factor.reshape(third_factor.shape[0], 1) * np.outer(r_0_pow_split[p_j], r_exp_filtered_curr_split[p_i][ts_for_pop_start + 6, :])

            r_cross_products = np.stack((
                r_exp_r_1_r_exp_sum,
                r_1_r_exp_r_exp_sum,

                r_exp_r_1_r_exp_sum_w,
                r_1_r_exp_r_exp_sum_w,
            ))

            w_updates_unweighted.append(r_cross_products)
            num_rules = w_updates_unweighted[-1].shape[0]
            w_updates_unweighted[-1][int(0.5 * num_rules):num_rules] = w_copy[pop_slices[p_j], pop_slices[p_i]] * w_updates_unweighted[-1][int(0.5 * num_rules):num_rules]

        coefs_0_pair = plasticity_coefs[coefficient_for_division_pair[0]]
        dw_hd_hd_unsummed = coefs_0_pair.reshape(coefs_0_pair.shape[0], 1, 1) * (w_updates_unweighted[0] * w_plastic[:n_e_pool, :n_e_pool] * w_not_almost_zero[:n_e_pool, :n_e_pool])
        effects_hd_hd_delta = np.sum(np.abs(dw_hd_hd_unsummed), axis=1)
        effects_hd_hd_delta = np.sum(effects_hd_hd_delta, axis=1)
        effects[coefficient_for_division_pair[0]] += effects_hd_hd_delta

        coefs_1_pair = plasticity_coefs[coefficient_for_division_pair[1]]
        dw_hd_hr_unsummed = coefs_1_pair.reshape(coefs_1_pair.shape[0], 1, 1) * (w_updates_unweighted[1] * w_plastic[n_e_pool:n_e, :n_e_pool] * w_not_almost_zero[n_e_pool:n_e, :n_e_pool])
        effects_hd_hr_delta = np.sum(np.abs(dw_hd_hr_unsummed), axis=1)
        effects_hd_hr_delta = np.sum(effects_hd_hr_delta, axis=1)
        effects[coefficient_for_division_pair[1]] += effects_hd_hr_delta

        coefs_2_pair = plasticity_coefs[coefficient_for_division_pair[2]]
        dw_hr_hd_unsummed = coefs_2_pair.reshape(coefs_2_pair.shape[0], 1, 1) * (w_updates_unweighted[2] * w_plastic[:n_e_pool, n_e_pool:n_e] * w_not_almost_zero[:n_e_pool, n_e_pool:n_e])
        effects_hr_hd_delta = np.sum(np.abs(dw_hr_hd_unsummed), axis=1)
        effects_hr_hd_delta = np.sum(effects_hr_hd_delta, axis=1)
        effects[coefficient_for_division_pair[2]] += effects_hr_hd_delta

        coefs_0_trip = plasticity_coefs[coefficient_for_division_trip[0]]
        dw_hd_hd_unsummed_trip = coefs_0_trip.reshape(coefs_0_trip.shape[0], 1, 1) * (w_updates_unweighted[3] * w_plastic[:n_e_pool, :n_e_pool] * w_not_almost_zero[:n_e_pool, :n_e_pool])
        effects_hd_hd_trip_delta = np.sum(np.abs(dw_hd_hd_unsummed_trip), axis=1)
        effects_hd_hd_trip_delta = np.sum(effects_hd_hd_trip_delta, axis=1)
        effects[coefficient_for_division_trip[0]] += effects_hd_hd_trip_delta

        coefs_1_trip = plasticity_coefs[coefficient_for_division_trip[1]]
        dw_hr_hd_unsummed_trip = coefs_1_trip.reshape(coefs_1_trip.shape[0], 1, 1) * (w_updates_unweighted[4] * w_plastic[:n_e_pool, n_e_pool:n_e] * w_not_almost_zero[:n_e_pool, n_e_pool:n_e])
        effects_hr_hd_trip_delta = np.sum(np.abs(dw_hr_hd_unsummed_trip), axis=1)
        effects_hr_hd_trip_delta = np.sum(effects_hr_hd_trip_delta, axis=1)
        effects[coefficient_for_division_trip[1]] += effects_hr_hd_trip_delta


        dw_hd_hd = np.sum(dw_hd_hd_unsummed, axis=0) + np.sum(dw_hd_hd_unsummed_trip, axis=0)
        dw_hd_hr = np.sum(dw_hd_hr_unsummed, axis=0)
        dw_hr_hd = np.sum(dw_hr_hd_unsummed, axis=0) + np.sum(dw_hr_hd_unsummed_trip, axis=0)

        w_copy[:n_e_pool, :n_e_pool] += (0.0005 * dw_hd_hd)
        w_copy[n_e_pool:n_e, :n_e_pool] += (0.0005 * dw_hd_hr)
        w_copy[:n_e_pool, n_e_pool:n_e] += (0.0005 * dw_hr_hd)

        # if sign of weight is flipped by update, set it to an infinitesimal amount with its initial polarity
        polarity_flip = sign_w * w_copy
        w_copy = np.where(polarity_flip >= 0, w_copy, inf_w)

    return w_copy, effects

