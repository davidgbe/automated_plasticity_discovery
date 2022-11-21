import numpy as np
from copy import deepcopy as copy

### For initiating activity

def generate_gaussian_pulse(t, u, s, w=1):
    return w / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * np.square((t-u) / s))

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

def simulate(t : np.ndarray, n_e : int, n_i : int, inp : np.ndarray, transfer_e, transfer_i, plasticity_coefs : np.ndarray, w : np.ndarray, w_plastic : np.ndarray, tau_e=5e-3, tau_i=5e-3, tau_stdp=25e-3, dt=1e-6, g=1, w_u=1, w_scale_factor=1.):    
    len_t = len(t)

    inh_activity = np.zeros((len_t))
    r = np.zeros((len_t, n_e + n_i))
    s = np.zeros((len_t, n_e + n_i))
    v = np.zeros((len_t, n_e + n_i))
    r_exp_filtered = np.zeros((len_t, n_e + n_i))

    w_copy = copy(w)
    sign_w = np.where(w >= 0, 1, -1).astype(int)
    inf_w = np.where(sign_w >= 0, 1e-6, -1e-6)

    tau = np.concatenate([tau_e * np.ones(n_e), tau_i * np.ones(n_i)])

    n_params = len(plasticity_coefs)
    one_third_n_params = int(n_params / 3)

    for i in range(0, len(t) - 1):
        v[i+1, :] = w_u * inp[i, :] + np.dot(w_copy, r[i, :].T) # calculate input to synaptic conductance equation
        s[i+1, :] = s[i, :] + (v[i+1, :] - s[i, :]) * dt / tau # update synaptic conductance as exponential filter of input

        # firing rates are calculated as normalized synaptic conductances
        r[i+1, :n_e] = g * transfer_e(s[i, :n_e])
        r[i+1, n_e:] = g * transfer_i(s[i, n_e:])
        
        # calculate exponential filtered of firing rate to use for STDP-like plasticity rules
        r_exp_filtered[i+1, :] = r_exp_filtered[i, :] * (1 - dt / tau_stdp) + r[i, :] * (dt / tau_stdp)

        r_0_pow = np.ones(n_e + n_i)
        r_1_pow = r[i+1, :] / 0.1
        r_2_pow = np.square(r[i+1, :]) / 0.01

        # find outer products of zeroth, first, and second powers of firing rates to compute updates due to plasticity rules
        r_0_r_0 = np.outer(r_0_pow, r_0_pow)
        r_0_r_1 = np.outer(r_1_pow, r_0_pow)
        r_1_r_0 = r_0_r_1.T
        r_0_r_2 = np.outer(r_2_pow, r_0_pow)
        r_2_r_0 = r_0_r_2.T
        r_1_r_1 = np.outer(r_1_pow, r_1_pow)
        r_1_r_2 = np.outer(r_2_pow, r_1_pow)
        r_2_r_1 = r_1_r_2.T
        r_2_r_2 = np.outer(r_2_pow, r_2_pow)

        r_0_r_exp = np.outer(r_exp_filtered[i+1, :], r_0_pow) / tau_stdp
        r_1_r_exp = np.outer(r_exp_filtered[i+1, :], r_1_pow) / tau_stdp
        r_exp_r_0 = r_0_r_exp.T
        r_exp_r_1 = r_1_r_exp.T

        r_cross_products = np.stack([
        	r_0_r_0,
        	r_0_r_1,
        	# r_1_r_0,
        	r_0_r_2,
        	# r_2_r_0,
        	r_1_r_1,
        	r_1_r_2,
        	# r_2_r_1,
        	# r_2_r_2,
            r_0_r_exp,
            # r_1_r_exp,
            r_exp_r_0,
            r_exp_r_1,
        ])

        # multiply firing rate outer products by 1 and w to form rules that do and do not scale with synapse size to form all updates for all rules
       	w_updates_unweighted = np.concatenate([r_cross_products, w_copy / w_scale_factor * r_cross_products])
       	
        # dot updates due to all rules with coefficients for these rules and compute total weight updates. Do not update non-plastic weights.
        dw_e_e = np.sum(plasticity_coefs[:one_third_n_params].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[:, :n_e, :n_e], axis=0) * w_plastic[:n_e, :n_e]
        dw_e_i = np.sum(plasticity_coefs[one_third_n_params:2*one_third_n_params].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[:, n_e:, :n_e], axis=0) * w_plastic[n_e:, :n_e]
        dw_i_e = np.sum(plasticity_coefs[2 * one_third_n_params:].reshape(one_third_n_params, 1, 1) * w_updates_unweighted[:, :n_e, n_e:], axis=0) * w_plastic[:n_e, n_e:]

       	w_copy[:n_e, :n_e] += 0.0005 * dw_e_e
        w_copy[n_e:, :n_e] += 0.0005 * dw_e_i
        w_copy[:n_e, n_e:] += 0.0005 * dw_i_e

        # if sign of weight is flipped by update, set it to an infinitesimal amount with its initial polarity
        polarity_flip = sign_w * w_copy
        w_copy = np.where(polarity_flip >= 0, w_copy, inf_w)

    return r, s, v, w_copy

