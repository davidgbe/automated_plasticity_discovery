from copy import deepcopy as copy
import numpy as np
import os
import time
from tqdm import tqdm
from aux_funcs import jax_gaussian_if_under_val, start_timer, zero_pad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
import jax
import jax.numpy as jnp
import jax.random as jr
from sklearn.linear_model import LinearRegression
from csv_reader import read_csv
from csv_writer import write_csv
from rate_network import simulate, calc_r_from_s, inv_softplus, softplus
from viz import plot_heatmap


### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=1, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--asp', metavar='asp', type=float, help='', default=0.)
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--load_initial', metavar='li', type=str, help='File from which to load the best params as an initial guess')
parser.add_argument('--frac_inputs_fixed', metavar='fi', type=float)
parser.add_argument('--syn_change_prob', metavar='cp', type=float, default=0.)
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--hd_hd_sparsity', metavar='dds', type=float, default=1.)
parser.add_argument('--hd_hr_sparsity', metavar='drs', type=float, default=1.)
parser.add_argument('--struct_prior', metavar='sp', type=str, default='random')

args = parser.parse_args()
print(args)

SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP = 320 # Number of times to simulate network and plasticity rules per loss function evaluation
decoder_train_trial_nums = (280, 300)
decoder_test_trial_nums = (300, 320)
READOUTS_PER_TRIAL = 20
STD_EXPL = args.std_expl
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_LOSS_COEF = args.asp
CHANGE_PROB_PER_ITER = args.syn_change_prob #0.0007
FRAC_INPUTS_FIXED = args.frac_inputs_fixed
INPUT_RATE_PER_CELL = 1000
INPUT_BLOCK_DURATION = 5e-3
N_RULES = 60 + 16
N_TIMECONSTS = 36 + 32
ETA = 1

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
w_e_e = 9e-4 / DT * 0.1 / np.sqrt(n_e_pool)
w_pool_side = -3e-4 / DT * 0.1 / np.sqrt(n_e_pool)
w_side_pool = 9e-4 / DT * 0.1 / np.sqrt(n_e_side)
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


# Make directory for outputting simulations
if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
joined_l1 = '_'.join([str(p) for p in L1_PENALTIES])
out_dir = f'sims_out/int_preexist_n40_speed_test_{BATCH_SIZE}_STD_EXPL_{STD_EXPL}_FIXED_{FIXED_DATA}_L1_PENALTY_{joined_l1}_ACT_PEN_{args.asp}_CHANGEP_{CHANGE_PROB_PER_ITER}_FRACI_{FRAC_INPUTS_FIXED}_SEED_{SEED}_{time_stamp}'
os.mkdir(out_dir)

# Make subdirectory for outputting CMAES info
os.mkdir(os.path.join(out_dir, 'outcmaes'))

# Made CSVs for outputting train & test data
header = ['evals', 'loss'] + [f'true_loss_{i}' for i in np.arange(BATCH_SIZE)]
header += list(rule_names)
header += ['effect_means']
header += ['effect_stds']

train_data_path = os.path.join(out_dir, 'train_data.csv')
write_csv(train_data_path, header)

test_data_path = os.path.join(out_dir, 'test_data.csv')
write_csv(test_data_path, header)


def create_shift_matrix(size, k=1):
	w = np.zeros((size, size))
	if k >= 1:
		for k_p in np.arange(1, k+1):
			w += np.diag(np.ones((size - k_p,)), k=k_p)
			### Add to make into a ring structure
			# w[(size - k_p):, k - k_p] = 1

	elif k <= -1:
		for k_p in np.arange(1, -k+1):
			w += np.diag(np.ones((size - k_p,)), k=-k_p)
			### Add to make into a ring structure
			# w[-k - k_p, (size - k_p):] = 1
	return w


def create_shuffled_one_to_one(size):
	w = np.diag(np.ones((size)))
	x = np.arange(size).astype(int)
	order = copy(x)
	np.random.shuffle(order)
	w[order, :] = w[np.arange(size), :]
	return w


def transform_zero_mean(X):
	X_zero_mean = X - jnp.mean(X, axis=0)
	return X_zero_mean


def calc_loss(r_train, r_test, targets_train, targets_test):

	invalid = jnp.any(jnp.isnan(r_train)) | jnp.any(jnp.isnan(r_test))
	
	r_train_normed = transform_zero_mean(r_train)
	r_test_normed = transform_zero_mean(r_test)

	targets_train_normed = transform_zero_mean(targets_train)
	targets_test_normed = transform_zero_mean(targets_test)

	RtR = jnp.matmul(jnp.transpose(r_train_normed), r_train_normed)
	Rty = jnp.matmul(jnp.transpose(r_train_normed), targets_train_normed[:, None])

	w = jnp.linalg.solve(RtR, Rty)

	singular_matrices_detected = jnp.any(jnp.isnan(w)) | jnp.any(jnp.isinf(w))

	w_screened = jnp.where(singular_matrices_detected, 0, w)

	residual = jnp.square((targets_test_normed - (r_test_normed @ w_screened).squeeze(1))).sum()
	total = jnp.square(targets_test_normed).sum()

	return jnp.where(invalid, 10, residual / total)


@jax.jit
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
    ee_mask = jr.uniform(keys[0], (n_e_pool, n_e_pool)) < args.hd_hd_sparsity
    ee_weights = w_e_e * jr.uniform(keys[1], (n_e_pool, n_e_pool))
    w_initial = w_initial.at[:n_e_pool, :n_e_pool].set(jnp.where(ee_mask, ee_weights, 0))

	### For initializing a ring-like shape in the pool neurons

	# x = np.arange(n_e_pool) / n_e_pool
	# connectivity_scale = 0.075
	# exp_ring_connectivity = 4 * w_e_e * (np.exp(-x/connectivity_scale) + np.exp((x-1)/connectivity_scale))

	# for r_idx in np.arange(n_e_pool):
	# 	w_initial[r_idx:n_e_pool, r_idx] = exp_ring_connectivity[:(n_e_pool - r_idx)]
	# 	w_initial[0:r_idx, r_idx] = exp_ring_connectivity[(n_e_pool - r_idx):]

	# w_initial[:n_e_pool, :n_e_pool] = w_initial[:n_e_pool, :n_e_pool] * (.jr.normal(key, size=(n_e_pool, n_e_pool)) * 0.1 + 1)

    # --- HR to HD connections ---
    if args.struct_prior == 'shift':
        shift_left = create_shift_matrix(n_e_side, k=3)
        shift_right = create_shift_matrix(n_e_side, k=-3)

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


def simulate_all(keys, X, train, track_params=True):
	np.random.seed(SEED)

	# c will have form like:
	# [
	# 	c_1 (key 1)
	# 	c_1 (key 2)
	# 	c_1 (key 3)
	#	c_2 (key 1)
	#   c_2 (key 2)
	#   c_2 (key 3)
	# ]

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

	m = np.abs(ws[0, ...]).max()
	plot_heatmap(ws[0, ...], cmap='bwr', vmin=-m, vmax=m, save_path='./figures/initial_matrix.png', figsize=(4, 3))

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

	train_size = (decoder_train_trial_nums[1] - decoder_train_trial_nums[0]) * READOUTS_PER_TRIAL
	test_size = (decoder_test_trial_nums[1] - decoder_test_trial_nums[0]) * READOUTS_PER_TRIAL

	r_train = np.empty((c.shape[0], train_size, n_e_pool))
	r_test = np.empty((c.shape[0], test_size, n_e_pool))

	targets_train = np.empty((c.shape[0], train_size))
	targets_test = np.empty((c.shape[0], test_size))

	train_idx = 0
	test_idx = 0

	for i in tqdm(range(N_INNER_LOOP)):
		timer = start_timer()

		r_in = np.empty((BATCH_SIZE, len(t), n_e_pool + 2 * n_e_side + n_i))
		running_input_sums = np.empty((BATCH_SIZE, INPUT_LEN))
		for i_k, key in enumerate(keys):
			r_in_k, running_input_sums_k = construct_inputs()
			r_in[i_k, :] = r_in_k
			running_input_sums[i_k, :] = running_input_sums_k

		r_in = jnp.tile(r_in, (len(X), *jnp.ones(r_in.ndim - 1).astype(int))) # duplicate block of inputs and integration targets by number of rules to test
		if i == 0:
			plot_heatmap(r_in[0, ...].T, cmap='hot', vmin=0, save_path='./figures/r_in_sample.png')

		train_trial_flag = (i >= decoder_train_trial_nums[0] and i < decoder_train_trial_nums[1])
		test_trial_flag = (i >= decoder_test_trial_nums[0] and i < decoder_test_trial_nums[1])

		if train_trial_flag or test_trial_flag:
			readout_times_for_trial = np.sort((np.random.rand(READOUTS_PER_TRIAL) * INPUT_LEN + INPUT_START) * DT)
			targets_for_readouts = running_input_sums[:, (readout_times_for_trial / DT).astype(int) - INPUT_START]
			targets_for_readouts = jnp.tile(targets_for_readouts, (len(X), *jnp.ones(targets_for_readouts.ndim - 1).astype(int)))
		else:
			readout_times_for_trial = np.array([])

		readout_times_for_trial = np.concatenate([readout_times_for_trial, np.array(t[-1:])])

		save_for_viewing = False # (i % 5 == 0)
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
			save_for_viewing=save_for_viewing
		)

		s, r_exp, inv_soft_w_all, syn, unstable = sol.ys

		W = softplus(inv_soft_w_all)

		ws = W[-1, ...]

		print(unstable)
		print(unstable.shape)
		print(jnp.sum(unstable))

		print('max W', jnp.abs(W).max())
		print('max s', s.max())

		print('max W', jnp.abs(W[:, ~unstable[-1, ...], ...]).max())
		print('max s', s[:, ~unstable[-1, ...], ...].max())

		
		inv_soft_w = inv_soft_w_all[-1, :]

		if i % 5 == 0 and i > 0:
			m = np.abs(ws[0, ...]).max()
			plot_heatmap(ws[0, ...], cmap='bwr', vmin=-m, vmax=m, save_path=f'./figures/weight_matrix_{zero_pad(i, 3)}.png', figsize=(4, 3))
			
			r = jnp.transpose(jax_calc_r(s, s_offsets, g, n_e), (1, 0, 2))
			print(r.shape)
			plot_heatmap(r[0, ...].T, cmap='hot', vmin=0, save_path=f'./figures/dynamics_{zero_pad(i, 3)}.png', figsize=(4, 3))

		# print('w abs summed', np.abs(ws).sum())
		print(syn.shape)
		print('syn', jnp.mean(syn[0, ...], axis=0))

		if train_trial_flag or test_trial_flag:
			if train_trial_flag:
				r_train[:, train_idx * READOUTS_PER_TRIAL : (train_idx + 1) * READOUTS_PER_TRIAL, :] = jnp.transpose(jax_calc_r(s[:-1, :, :n_e_pool], s_offsets[:n_e_pool], g, n_e), (1, 0, 2))
				targets_train[:, train_idx * READOUTS_PER_TRIAL : (train_idx + 1) * READOUTS_PER_TRIAL] = targets_for_readouts
				train_idx += 1
			else:
				r_test[:, test_idx * READOUTS_PER_TRIAL : (test_idx + 1) * READOUTS_PER_TRIAL, :] = jnp.transpose(jax_calc_r(s[:-1, :, :n_e_pool], s_offsets[:n_e_pool], g, n_e), (1, 0, 2))
				targets_test[:, test_idx * READOUTS_PER_TRIAL : (test_idx + 1) * READOUTS_PER_TRIAL] = targets_for_readouts
				test_idx += 1
		timer()

	jax_calc_loss = jax.vmap(calc_loss, (0, 0, 0, 0))
	losses = jax_calc_loss(r_train, r_test, targets_train, targets_test)
	print('raw losses')
	print(losses)
	losses_for_coefs = 1000 * jnp.reshape(losses, (len(X), keys.shape[0])).mean(axis=1)
	return losses_for_coefs


def log_sim_results(write_path, eval_tracker, loss, true_losses, plasticity_coefs, syn_effects):
	# eval_num, loss, true_losses, plastic_coefs, syn_effects
	syn_effect_means = np.mean(syn_effects, axis=0)
	syn_effect_stds = np.std(syn_effects, axis=0)
	to_save = np.concatenate([[eval_tracker['evals'], loss], true_losses, plasticity_coefs, syn_effect_means, syn_effect_stds]).flatten()
	print(to_save)
	write_csv(write_path, list(to_save))


def process_plasticity_rule_results(results, x, eval_tracker=None, train=True):
	plasticity_coefs = x[:N_RULES]
	rule_time_constants = x[N_RULES:]

	if np.any(np.array([res['blew_up'] for res in results])):
		if eval_tracker is not None:
			eval_tracker['evals'] += 1
		return 1e8 * BATCH_SIZE + 1e7 * np.sum(np.abs(plasticity_coefs)), 1e8 * np.ones((len(results),)), np.zeros((len(results), len(plasticity_coefs))), np.zeros((len(results),)), np.zeros((len(results),))

	true_losses = np.array([res['loss'] for res in results])
	syn_effects = np.stack([res['syn_effects'] for res in results])
	total_activity_penalties = ACTIVITY_LOSS_COEF * np.array([res['rs_for_loss'].mean() for res in results])
	syn_effect_penalties = L1_PENALTIES[0] * np.sum(np.abs(syn_effects), axis=1)

	losses = true_losses + syn_effect_penalties + total_activity_penalties
	loss = np.sum(losses)

	if eval_tracker is not None:
		if train:
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
				if eval_tracker['evals'] > 0:
					eval_tracker['best_loss'] = loss
					eval_tracker['best_changed'] = True
					eval_tracker['params'] = copy(x)

				plot_results(
					results,
					eval_tracker,
					out_dir,
					plasticity_coefs,
					true_losses,
					syn_effect_penalties,
					total_activity_penalties,
					train=True,
				)
			eval_tracker['evals'] += 1
		else:
			plot_results(
				results,
				eval_tracker,
				out_dir,
				plasticity_coefs,
				true_losses,
				syn_effect_penalties,
				total_activity_penalties,
				train=False,
			)
			eval_tracker['best_changed'] = False

	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')
	return loss, true_losses, syn_effects, syn_effect_penalties, total_activity_penalties


def load_best_params(file_name):
	file_path = f'./sims_out/{file_name}/outcmaes/xrecentbest.dat'
	df_params = read_csv(file_path, read_header=False)
	x = np.arange(df_params.shape[0])
	min_loss_idx = np.argmin([df_params.iloc[i][4] for i in x])
	best_params = df_params.iloc[min_loss_idx][5:]
	return np.array(best_params)

	# # make this into jax
	# simulate_all(task_vars)

	# losses = []
	# for i in range(len(X)):
	# 	loss, true_losses, syn_effects, syn_effect_penalties, total_activities = process_plasticity_rule_results(results[BATCH_SIZE * i: BATCH_SIZE * (i+1)], X[i], eval_tracker=eval_tracker, train=train)
	# 	losses.append(loss)
	# 	if train:
	# 		log_sim_results(train_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
	# 	else:
	# 		log_sim_results(test_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
	
	# dur = time.time() - start
	# print('dur:', dur)

	# return losses

def process_params_str(s):
	params = []
	for x in s.split(' '):
		x = x.replace('\n', '')
		if x != '':
			params.append(float(x))
	return np.array(params)

if __name__ == '__main__':
	mp.set_start_method('fork')

	if args.load_initial is not None:
		x0 = load_best_params(args.load_initial)
	else:
		x0 = np.concatenate([np.zeros(N_RULES), 5e-3 * np.ones(N_TIMECONSTS)])

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
	}

	# eval_all([x0], eval_tracker=eval_tracker)

	options = {
		'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
		'popsize': 30,
		'bounds': [
			[-10] * N_RULES + [0.5e-3] * N_TIMECONSTS,
			[10] * N_RULES + [40e-3] * N_TIMECONSTS,
		],
	}

	es = cma.CMAEvolutionStrategy(x0, STD_EXPL, options)
	options['popsize'] = es.opts['popsize']

	# eval_all([x0], eval_tracker=eval_tracker, train=False)

	key = jr.key(0)
	keys = jr.split(key, len(train_seeds))

	# X0 = [x0]
	# base_losses = simulate_all(keys, X0, True, track_params=True)
	# print(base_losses)

	while not es.stop():
		X = es.ask()
		losses = simulate_all(keys, X, True, track_params=True)
		print(losses)
		print(losses.shape)
		es.tell(X, losses.tolist())
		# if eval_tracker['best_changed']:
		# 	eval_all([eval_tracker['params']], eval_tracker=eval_tracker, train=False)
		# es.disp()
