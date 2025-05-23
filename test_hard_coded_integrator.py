from copy import deepcopy as copy
import numpy as np
import os
import time
from tqdm import tqdm
from aux_funcs import jax_gaussian_if_under_val, start_timer, zero_pad, find_dirs_with_fragment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
from sklearn.linear_model import LinearRegression
from csv_reader import read_csv
from csv_writer import write_csv
from rate_network import simulate, calc_r_from_s, inv_softplus, softplus
from viz import plot_heatmap, format_plot


### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--eta', metavar='et', type=float, help='Const coefficient that sets scale of learning rates relative to "std_expl"')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=1, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--activity_penalty', metavar='ap', type=float, help='', default=0.)
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--load_initial', metavar='li', type=str, help='File from which to load the best params as an initial guess')
parser.add_argument('--frac_inputs_fixed', metavar='fi', type=float)
parser.add_argument('--syn_change_prob', metavar='cp', type=float, default=0.)
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--hd_hd_sparsity', metavar='dds', type=float, default=1.)
parser.add_argument('--hd_hr_sparsity', metavar='drs', type=float, default=1.)
parser.add_argument('--struct_prior', metavar='sp', type=str, default='random')
parser.add_argument('--run_num', metavar='rn', type=int)

args = parser.parse_args()
print(args)

RUN_NUM = zero_pad(args.run_num, 6)
BATCH_SIZE = args.batch
SEED = args.seed
TEST_SEED = SEED + 2 * BATCH_SIZE
N_INNER_LOOP = 40 # Number of times to simulate network and plasticity rules per loss function evaluation
decoder_train_trial_nums = (0, 20)
decoder_test_trial_nums = (20, 40)
READOUTS_PER_TRIAL = 20
STD_EXPL = args.std_expl
ETA = args.eta
DW_LAG = 5
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_PENALTY = args.activity_penalty
CHANGE_PROB_PER_ITER = args.syn_change_prob #0.0007
FRAC_INPUTS_FIXED = args.frac_inputs_fixed
INPUT_RATE_PER_CELL = 1000
INPUT_BLOCK_DURATION = 5e-3
N_RULES = 60 + 16
N_TIMECONSTS = 36 + 32

T = 0.1 # Total duration of one network simulation
DT = 1e-4 # Timestep
INPUT_START = int(20e-3/DT)
INPUT_END = int(100e-3/DT)
INPUT_LEN = INPUT_END - INPUT_START
input_block_timesteps = int(INPUT_BLOCK_DURATION / DT)

t = np.linspace(0, T, int(T / DT))
n_e_pool = 40 # Number excitatory cells in sequence (also length of sequence)
n_e_side = 40
n_e = n_e_pool + 2 * n_e_side
n_i = 1 # Number inhibitory cells
train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)

# scaling these 3 parameters by a factor 10 up will give appropriate values for ring attracting circuit
w_e_e = 9e-4 / DT / np.sqrt(n_e_pool)
w_pool_side = -3e-4 / DT / np.sqrt(n_e_pool)
w_side_pool = 9e-4 / DT / np.sqrt(n_e_side)
w_e_i = 2.5e-4 / DT / np.sqrt(n_e_pool)
w_i_e = -1e-4 / DT / np.sqrt(n_i)
TAU_E = 5e-3
TAU_I = 0.1e-3
TAU_ALPHA_INPUT = 3e-3

s_offsets = jnp.concatenate((jnp.full(n_e, 0.1), jnp.full(n_i, 0)))
g = 1
w_u = 1
tau_s = jnp.concatenate((jnp.full(n_e, TAU_E), jnp.full(n_i, TAU_I)))

rule_names = [ # Define labels for all rules to be run during simulations
    r'',
    r'$y$',
    r'$x$',
    r'$x \, y$',
    r'$\tilde{y}$',
    r'$x \, \tilde{y}$',
    r'$\tilde{x}$',
    r'$\tilde{x} \, y$',
    r'$\tilde{y} \, y$',
    r'$\tilde{x} \, x$',

    r'$w$',
    r'$w y$',
    r'$w x$',
    r'$w x \, y$',
    r'$w \tilde{y}$',
    r'$w x \, \tilde{y}$',
    r'$w \tilde{x}$',
    r'$w \tilde{x} \, y$',
    r'$w \tilde{y} \, y$',
    r'$w \tilde{x} \, x$',
]

rule_names = [
    [r'$HD \rightarrow HD$ ' + r_name for r_name in rule_names],
    [r'$HD \rightarrow HR$ ' + r_name for r_name in rule_names],
    [r'$HR \rightarrow HD$ ' + r_name for r_name in rule_names],
]

rule_names_tripartite = [
    r'$\tilde{y}, z \sim y$',
    r'$\tilde{x}, z \sim y$',
    r'$x \tilde{y}, z \sim y$',
    r'$\tilde{x} y, z \sim y$',

    r'$w \tilde{y}, z \sim y$',
    r'$w \tilde{x}, z \sim y$',
    r'$w x \tilde{y}, z \sim y$',
    r'$w \tilde{x} y, z \sim y$',
]

rule_names += [
    [r'$HD \rightarrow HD, HR \sim HD$' + r_name for r_name in rule_names_tripartite],
    [r'$HR \rightarrow HD, HD \sim HD$' + r_name for r_name in rule_names_tripartite],
]


rule_names = [r for rs in rule_names for r in rs]
rule_names = np.array(rule_names, dtype=object)


def create_shift_matrix(size, k=1, ring=False):
    w = np.zeros((size, size))
    if k >= 1:
        for k_p in np.arange(1, k+1):
            w += np.diag(np.ones((size - k_p,)), k=k_p)
            ### Add to make into a ring structure
            if ring:
                w[(size - k_p):, k - k_p] = 1

    elif k <= -1:
        for k_p in np.arange(1, -k+1):
            w += np.diag(np.ones((size - k_p,)), k=-k_p)
            ### Add to make into a ring structure
            if ring:
                w[-k - k_p, (size - k_p):] = 1
    return jnp.array(w)


def create_shuffled_one_to_one(size):
    w = np.diag(np.ones((size)))
    x = np.arange(size).astype(int)
    order = copy(x)
    np.random.shuffle(order)
    w[order, :] = w[np.arange(size), :]
    return w


def transform_zero_mean(X):
    X_mean = jnp.mean(X, axis=0)
    X_zero_mean = X - X_mean
    return X_zero_mean, X_mean


def calc_loss(r_train, r_test, targets_train, targets_test):

    invalid = jnp.any(jnp.isnan(r_train)) | jnp.any(jnp.isnan(r_test))
    
    r_train_normed, r_train_mean = transform_zero_mean(r_train)
    r_test_normed = r_test - r_train_mean

    targets_train_normed, targets_train_mean = transform_zero_mean(targets_train)
    targets_test_normed = targets_test - targets_train_mean

    RtR = jnp.matmul(jnp.transpose(r_train_normed), r_train_normed)
    Rty = jnp.matmul(jnp.transpose(r_train_normed), targets_train_normed[:, None])

    w = jnp.linalg.solve(RtR, Rty)

    singular_matrices_detected = jnp.any(jnp.isnan(w)) | jnp.any(jnp.isinf(w))

    w_screened = jnp.where(singular_matrices_detected, 0, w)

    residual = jnp.square((targets_test_normed - (r_test_normed @ w_screened).squeeze(1))).sum()
    total = jnp.square(targets_test_normed).sum()

    return jnp.where(invalid, 10, residual / total)


def make_network(key):
    '''
    Generates an excitatory chain with recurrent inhibition and weak recurrent excitation.
    Weights that form sequence are distorted randomly.
    '''
    total_size = n_e_pool + 2 * n_e_side + n_i
    w_initial = jnp.zeros((total_size, total_size))

    # Pre-split all keys
    num_keys = 15  # You may need to increase if more randomness is added
    key, *keys = jr.split(key, num_keys)
    
    # --- E to E (sparse random) ---
    ee_mask = jr.uniform(keys[0], (n_e_pool, n_e_pool)) <= args.hd_hd_sparsity

    ### For initializing a ring-like shape in the pool neurons
    connectivity_scale = 2

    shift_mats = []

    for i in range(1, 10):
        w_shift = jnp.diag(jnp.ones(n_e_pool - jnp.abs(i)), k=i) * w_e_e * jnp.exp(-jnp.abs(i) / connectivity_scale)
        shift_mats.append(w_shift)
        shift_mats.append(jnp.transpose(w_shift))

    w_pool_pool = jnp.sum(jnp.stack(shift_mats), axis=0)
    w_initial = w_initial.at[:n_e_pool, :n_e_pool].set(jnp.where(ee_mask, w_pool_pool, 0))

    # --- HR to HD connections ---
    if args.struct_prior == 'shift':
        shift_left = create_shift_matrix(n_e_side, k=5)
        shift_right = create_shift_matrix(n_e_side, k=-5)

        left_mask = jr.uniform(keys[2], (n_e_pool, n_e_side)) < args.hd_hr_sparsity
        right_mask = jr.uniform(keys[3], (n_e_pool, n_e_side)) < args.hd_hr_sparsity

        w_initial = w_initial.at[:n_e_pool, n_e_pool:n_e_pool + n_e_side].set(
            w_side_pool * jnp.where(left_mask, shift_left, 0)
        )
        w_initial = w_initial.at[:n_e_pool, n_e_pool + n_e_side:n_e_pool + 2 * n_e_side].set(
            w_side_pool * jnp.where(right_mask, shift_right, 0)
        )

        # Inhibitory backward connections from HD to HR
        input_template = w_pool_side * (1 - (shift_left + shift_right))
        input_template = input_template.at[jnp.diag_indices(n_e_side)].set(0)

        back_mask_L = jr.uniform(keys[4], (n_e_side, n_e_pool)) < args.hd_hr_sparsity
        back_mask_R = jr.uniform(keys[5], (n_e_side, n_e_pool)) < args.hd_hr_sparsity

        w_initial = w_initial.at[n_e_pool:n_e_pool + n_e_side, :n_e_pool].set(
            jnp.where(back_mask_L, input_template, 0)
        )
        w_initial = w_initial.at[n_e_pool + n_e_side:n_e_pool + 2 * n_e_side, :n_e_pool].set(
            jnp.where(back_mask_R, input_template, 0)
        )

    else:
        # Random sparse connections instead of structured shift
        for i in range(2):  # left and right
            mask = jr.uniform(keys[2 + i], (n_e_pool, n_e_side)) < args.hd_hr_sparsity
            weights = jr.uniform(keys[4 + i], (n_e_pool, n_e_side))
            w_initial = w_initial.at[
                :n_e_pool, n_e_pool + i * n_e_side:n_e_pool + (i + 1) * n_e_side
            ].set(w_side_pool * jnp.where(mask, weights, 0))

        for i in range(2):  # back to pool
            mask = jr.uniform(keys[6 + i], (n_e_side, n_e_pool)) < args.hd_hr_sparsity
            weights = jr.uniform(keys[8 + i], (n_e_side, n_e_pool))
            w_initial = w_initial.at[
                n_e_pool + i * n_e_side:n_e_pool + (i + 1) * n_e_side, :n_e_pool
            ].set(w_pool_side * jnp.where(mask, weights, 0))

    # --- E to I ---
    ei_weights = jax_gaussian_if_under_val(keys[10], 1, (n_i, n_e_pool), w_e_i, 0 * w_e_i)
    w_initial = w_initial.at[-n_i:, :n_e_pool].set(ei_weights)

    # --- I to E ---
    ie_weights = jax_gaussian_if_under_val(keys[11], 1, (n_e_pool, n_i), w_i_e, 0 * jnp.abs(w_i_e))
    w_initial = w_initial.at[:n_e_pool, -n_i:].set(ie_weights)

    # --- Zero out diagonal ---
    w_initial = w_initial.at[jnp.diag_indices(total_size)].set(0)

    return w_initial


def calc_alpha_func(tau_alpha):
    alpha_func_n_steps = int(10 * tau_alpha / DT)
    t_alpha = np.arange(0, alpha_func_n_steps) * DT
    return np.e * t_alpha / tau_alpha * np.exp(-t_alpha/tau_alpha)


def poisson_arrivals_to_inputs(arrivals, tau_alpha):
    alpha_func = calc_alpha_func(tau_alpha)
    input_current = np.zeros(arrivals.shape)

    for i in range(arrivals.shape[1]):
        input_current[:, i] = np.convolve(alpha_func, arrivals[:, i], mode='full')[:arrivals.shape[0]]
    return input_current


def construct_inputs(input_size=6):
    input_spks = np.zeros((INPUT_LEN, 2 * n_e_side))
    inputs = np.zeros((INPUT_LEN,)).astype(int)
    inputs[0] = 1

    for k in range(INPUT_LEN):
        if k % input_block_timesteps == 0:
            input_state = np.random.choice([-1, 0, 1])
            inputs[k] = input_state
            if input_state != 0:
                input_block = np.random.poisson(lam=2 * INPUT_RATE_PER_CELL * DT, size=(input_block_timesteps, n_e_side))
                if input_state == -1:
                    input_spks[k : k + input_block_timesteps, :n_e_side] = input_block
                elif input_state == 1:
                    input_spks[k : k + input_block_timesteps, n_e_side : 2 * n_e_side] = input_block
        else:
            inputs[k] = inputs[k-1]
        
    filtered_input_to_sum_per_neuron = poisson_arrivals_to_inputs(input_spks, TAU_ALPHA_INPUT)
    filtered_input_to_sum = filtered_input_to_sum_per_neuron[:, n_e_side:2 * n_e_side].sum(axis=1) - filtered_input_to_sum_per_neuron[:, :n_e_side].sum(axis=1)
    running_input_sums = np.cumsum(filtered_input_to_sum) / INPUT_LEN
    
    r_in_spks = np.zeros((len(t), n_e_pool + 2 * n_e_side + n_i))
    
    input_slice = (int((n_e_pool - input_size)/ 2), int((n_e_pool + input_size)/ 2))
    r_in_spks[:int(10e-3/DT), input_slice[0]:input_slice[1]] = np.random.poisson(lam=INPUT_RATE_PER_CELL * DT, size=(int(10e-3/DT), input_size))
    
    r_in_spks[INPUT_START:INPUT_END, n_e_pool:n_e_pool + 2 * n_e_side] = input_spks
    r_in = poisson_arrivals_to_inputs(r_in_spks, TAU_ALPHA_INPUT)

    r_in[:, :n_e_pool] = 0.25 * r_in[:, :n_e_pool]
    r_in[:, n_e_pool:(n_e_pool + 2 * n_e_side)] = 0.1 * r_in[:, n_e_pool:(n_e_pool + 2 * n_e_side)]

    r_in[:, :n_e_pool] += (
        0.02 * poisson_arrivals_to_inputs(np.random.poisson(lam=INPUT_RATE_PER_CELL * DT, size=(len(t), n_e_pool)), TAU_ALPHA_INPUT)
    )

    return r_in, running_input_sums


def batchify(x, n_batch):
    x = jnp.array(x)
    return jnp.tile(x, (n_batch, *jnp.ones(x.ndim).astype(int)))


jax_calc_r = jax.vmap(
    jax.vmap(calc_r_from_s, (0, None, None, None)),
    (0, None, None, None)
)


def simulate_all(all_keys, X, train, eval_tracker):
    if train:
        np.random.seed(SEED)
    else:
        np.random.seed(TEST_SEED)

    # c will have form like:
    # [
    # 	c_1 (key 1)
    # 	c_1 (key 2)
    # 	c_1 (key 3)
    #	c_2 (key 1)
    #   c_2 (key 2)
    #   c_2 (key 3)
    # ]

    if train:
        keys = all_keys[:BATCH_SIZE]
    else:
        keys = all_keys[BATCH_SIZE:]

    c = jnp.concatenate([
        jnp.tile(jnp.array(x[:N_RULES]), (keys.shape[0], 1))
        for x in X
    ])

    tau_rules = jnp.concatenate([
        jnp.tile(jnp.array(x[N_RULES:]), (keys.shape[0], 1))
        for x in X
    ])

    ws_base = jax.vmap(make_network, (0,))(keys) # generate a weight matrix for each key
    inv_soft_w_base = inv_softplus(ws_base)
    ws_polarity_base = jnp.where(ws_base >= 0, 1, -1)
    ws_nonzero_base = jnp.where(ws_base != 0, 1, 0).astype(int)

    inv_soft_w = jnp.tile(inv_soft_w_base, (len(X), *jnp.ones(inv_soft_w_base.ndim - 1).astype(int))) # duplicate the block of all weight matrices for the number of rules 
    ws_polarity = jnp.tile(ws_polarity_base, (len(X), *jnp.ones(ws_polarity_base.ndim - 1).astype(int)))
    ws_nonzero = jnp.tile(ws_nonzero_base, (len(X), *jnp.ones(ws_nonzero_base.ndim - 1).astype(int))) 

    ws = softplus(inv_soft_w) * ws_polarity * ws_nonzero

    if eval_tracker['evals'] == 0:
        m = np.abs(ws[0, ...]).max()
        save_path = os.path.join(out_dir, 'initial_matrix.png')
        plot_heatmap(
            matrix=ws[0, ...],
            cmap='bwr',
            vmin=-m,
            vmax=m,
            save_path=save_path,
            figsize=(4, 3),
            ylabel='Neuron index',
            xlabel='Neuron index',
            title=None,
        )

    args = (
        c,
        tau_rules,
        g,
        s_offsets,
        w_u,
        tau_s,
        ETA,
        n_e,
        n_i,
        n_e_pool,
        n_e_side,
    )

    total_abs_synaptic_change = np.zeros((c.shape[0], N_RULES))

    train_size = (decoder_train_trial_nums[1] - decoder_train_trial_nums[0]) * READOUTS_PER_TRIAL
    test_size = (decoder_test_trial_nums[1] - decoder_test_trial_nums[0]) * READOUTS_PER_TRIAL

    r_train = np.empty((c.shape[0], train_size, n_e_pool))
    r_test = np.empty((c.shape[0], test_size, n_e_pool))

    targets_train = np.empty((c.shape[0], train_size))
    targets_test = np.empty((c.shape[0], test_size))

    train_idx = 0
    test_idx = 0

    all_rs_for_viz = np.empty((c.shape[0], int((train_size + test_size) / READOUTS_PER_TRIAL), 1000, n_e + n_i)) # (batch_index, activation_index, T, neurons)

    for i in tqdm(range(N_INNER_LOOP)):

        r_in = np.empty((BATCH_SIZE, len(t), n_e_pool + 2 * n_e_side + n_i))
        running_input_sums = np.empty((BATCH_SIZE, INPUT_LEN))
        for i_k, key in enumerate(keys):
            r_in_k, running_input_sums_k = construct_inputs()
            r_in[i_k, :] = r_in_k
            running_input_sums[i_k, :] = running_input_sums_k

        r_in = jnp.tile(r_in, (len(X), *jnp.ones(r_in.ndim - 1).astype(int))) # duplicate block of inputs and integration targets by number of rules to test
        
        if i == 0 and eval_tracker['evals'] == 0:
            save_path = os.path.join(out_dir, 'r_in_sample.png')
            plot_heatmap(r_in[0, ...].T, cmap='hot', vmin=0, save_path=save_path)

        train_trial_flag = (i >= decoder_train_trial_nums[0] and i < decoder_train_trial_nums[1])
        test_trial_flag = (i >= decoder_test_trial_nums[0] and i < decoder_test_trial_nums[1])

        if train_trial_flag or test_trial_flag:
            readout_times_for_trial = np.sort((np.random.rand(READOUTS_PER_TRIAL) * INPUT_LEN + INPUT_START) * DT)
            targets_for_readouts = running_input_sums[:, (readout_times_for_trial / DT).astype(int) - INPUT_START]
            targets_for_readouts = jnp.tile(targets_for_readouts, (len(X), *jnp.ones(targets_for_readouts.ndim - 1).astype(int)))
        else:
            readout_times_for_trial = np.array([])

        if train:
            readout_times_for_trial = np.concatenate([readout_times_for_trial, np.array(t[-1:])])


        sol = simulate(
            t,
            inv_soft_w,
            ws_polarity,
            ws_nonzero,
            r_in,
            c,
            tau_rules,
            n_e + n_i,
            DT,
            readout_times_for_trial,
            args,
            save_for_viewing=not train
        )

        s, r_exp, inv_soft_w_all, syn, unstable = sol

        inv_soft_w = inv_soft_w_all[-1, :]
        final_synaptic_change = syn[-1, ...]
        total_abs_synaptic_change += final_synaptic_change

        if train_trial_flag or test_trial_flag:
            r = jnp.transpose(jax_calc_r(s, s_offsets, g, n_e), (1, 0, 2))

            if not train:
                saved_activation_num = i - decoder_train_trial_nums[0]
                all_rs_for_viz[:, saved_activation_num, ...] = r[:, READOUTS_PER_TRIAL:, :] # (batch_index, activation_index, T, neurons)

            if train_trial_flag:
                r_train[:, train_idx * READOUTS_PER_TRIAL : (train_idx + 1) * READOUTS_PER_TRIAL, :n_e_pool] = r[:, :READOUTS_PER_TRIAL, :n_e_pool]
                targets_train[:, train_idx * READOUTS_PER_TRIAL : (train_idx + 1) * READOUTS_PER_TRIAL] = targets_for_readouts
                train_idx += 1
            else:
                r_test[:, test_idx * READOUTS_PER_TRIAL : (test_idx + 1) * READOUTS_PER_TRIAL, :n_e_pool] = r[:, :READOUTS_PER_TRIAL, :n_e_pool]
                targets_test[:, test_idx * READOUTS_PER_TRIAL : (test_idx + 1) * READOUTS_PER_TRIAL] = targets_for_readouts
                test_idx += 1

    jax_calc_loss = jax.vmap(calc_loss, (0, 0, 0, 0))
    losses = jax_calc_loss(r_train, r_test, targets_train, targets_test)
    final_instability = unstable[-1, :]
    losses = jnp.where(final_instability > 0, 1e7, losses)

    losses_for_coefs = 1000 * jnp.reshape(losses, (len(X), keys.shape[0])).mean(axis=1)
    syn_effects_for_coefs = jnp.reshape(total_abs_synaptic_change, (len(X), keys.shape[0], total_abs_synaptic_change.shape[1])).mean(axis=1)

    print('losses')
    print(losses_for_coefs)

    write_path = train_data_path if train else test_data_path
    log_results(write_path, eval_tracker, losses_for_coefs, jnp.array(X), syn_effects_for_coefs)

    min_loss_index = np.argmin(losses_for_coefs)
    if train:
        eval_tracker['evals'] += 1
        if (losses_for_coefs[min_loss_index] < eval_tracker['best_loss']):
            eval_tracker['best_x'] = X[min_loss_index]
            eval_tracker['best_loss'] = losses_for_coefs[min_loss_index]
            eval_tracker['best_changed'] = True
    else:
        W = softplus(inv_soft_w_all) * ws_polarity * ws_nonzero
        ws = W[-1, ...]

        plot_run(losses, ws, all_rs_for_viz, eval_tracker)
        eval_tracker['best_changed'] = False

    return losses_for_coefs


def log_results(write_path, eval_tracker, losses, plasticity_coefs, syn_effects):
    # eval_num, loss, true_losses, plastic_coefs, syn_effects
    evals = np.full((losses.shape[0], 1), eval_tracker['evals']) 
    all_save_data = np.concatenate([evals, losses.reshape((losses.shape[0], 1)), plasticity_coefs, syn_effects], axis=1)
    for i in range(all_save_data.shape[0]):
        save_data = all_save_data[i, :]
        write_csv(write_path, list(save_data))


def plot_run(losses, ws, all_rs_for_viz, eval_tracker):
    padded_idx = zero_pad(eval_tracker['evals'], 4)
    save_path = os.path.join(out_dir, f'{padded_idx}.png')

    scale = 2
    total_height = 3 * losses.shape[0] * scale
    total_width = 3 * scale

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(nrows=3 * losses.shape[0], ncols=2, width_ratios=[2, 1.25], height_ratios=[1]* 3 * losses.shape[0])
    axs = []

    for i in range(losses.shape[0]):
        ax_w = fig.add_subplot(gs[3 * i, 1])

        w = ws[i, ...]
        rs_for_trials = all_rs_for_viz[i, ...]

        m = np.abs(w).max()
        _, _, w_cbar  = plot_heatmap(
            matrix=w,
            ax=ax_w,
            cmap='bwr',
            vmin=-m,
            vmax=m,
            ylabel='Neuron index',
            xlabel='Neuron index',
            title=None,
        )
        format_plot(
            w_cbar.ax,
            ticklabelsize=8,
            axislabelsize=10,
        )

        for j in range(3):
            ax_r_j = fig.add_subplot(gs[3 * i + j, 0])
            _, _, cbar_j = plot_heatmap(rs_for_trials[-j, ...].T, ax_r_j, cmap='hot', vmin=0)
            format_plot(
                cbar_j.ax,
                ticklabelsize=8,
                axislabelsize=10,
            )

            if j == 0:
                axs.append([ax_r_j, ax_w])
            else:
                axs.append([ax_r_j])


    format_plot(
        [ax for row_ax in axs for ax in row_ax],
        ticklabelsize=8,
        axislabelsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.close()


def load_best_params(file_name):
    file_path = f'./sims_out/{file_name}/outcmaes/xrecentbest.dat'
    df_params = read_csv(file_path, read_header=False)
    x = np.arange(df_params.shape[0])
    min_loss_idx = np.argmin([df_params.iloc[i][4] for i in x])
    best_params = df_params.iloc[min_loss_idx][5:]
    return np.array(best_params)


def process_params_str(s):
    params = []
    for x in s.split(' '):
        x = x.replace('\n', '')
        if x != '':
            params.append(float(x))
    return np.array(params)

if __name__ == '__main__':
    mp.set_start_method('fork')

    # Make directory for outputting simulations
    if not os.path.exists('sims_out'):
        os.mkdir('sims_out')

    existing_dirs_with_run_num = find_dirs_with_fragment('sims_out', f'run_{RUN_NUM}')

    if len(existing_dirs_with_run_num) == 0: # no existing files
        # Make subdirectory for this particular experiment
        time_stamp = str(datetime.now()).replace(' ', '_')
        joined_l1 = '_'.join([str(p) for p in L1_PENALTIES])
        out_dir = f'sims_out/int_hard_coded_{BATCH_SIZE}_ETA_{ETA}_L1_PENALTY_{joined_l1}_ACT_PEN_{args.activity_penalty}_CHANGEP_{CHANGE_PROB_PER_ITER}_FRACI_{FRAC_INPUTS_FIXED}_SEED_{SEED}_{time_stamp}_run_{RUN_NUM}'
        os.mkdir(out_dir)

        # Make subdirectory for outputting CMAES info
        os.mkdir(os.path.join(out_dir, 'outcmaes'))

        # Made CSVs for outputting train & test data
        header = ['evals', 'loss']
        header += list(rule_names)

        train_data_path = os.path.join(out_dir, 'train_data.csv')
        write_csv(train_data_path, header)

        test_data_path = os.path.join(out_dir, 'test_data.csv')
        write_csv(test_data_path, header)

        eval_tracker = {
            'evals': 0,
            'best_loss': np.inf,
            'best_x': np.nan,
            'best_changed': False,
        }

        options = {
            'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
            'popsize': 30,
            'bounds': [
                [-10] * N_RULES + [0.5e-3] * N_TIMECONSTS,
                [10] * N_RULES + [40e-3] * N_TIMECONSTS,
            ],
        }

        if args.load_initial is not None:
            x0 = load_best_params(args.load_initial)
        else:
            x0 = np.concatenate([np.zeros(N_RULES), 5e-3 * np.ones(N_TIMECONSTS)])

        es = cma.CMAEvolutionStrategy(x0, STD_EXPL, options)
        options['popsize'] = es.opts['popsize']
    else:
        out_dir = os.path.join('sims_out', existing_dirs_with_run_num[-1])
        train_data_path = os.path.join(out_dir, 'train_data.csv')
        test_data_path = os.path.join(out_dir, 'test_data.csv')
        with open(os.path.join(out_dir, 'eval_tracker.pkl'), 'rb') as f:
            eval_tracker = pickle.load(f)
        with open(os.path.join(out_dir, 'es_checkpoint.pkl'), 'rb') as f:
            es = pickle.load(f)

    key = jr.key(SEED)
    keys = jr.split(key, 2 * BATCH_SIZE)

    base_losses = simulate_all(keys, [x0], False, eval_tracker)
    print(base_losses)
