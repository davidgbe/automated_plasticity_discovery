from copy import deepcopy as copy
import numpy as np
import os
import sys
import time
from functools import partial
from aux import gaussian_if_under_val, start_timer, zero_pad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
import numba
from scipy.sparse import csc_matrix
from sklearn.linear_model import LinearRegression
from csv_reader import read_csv
from csv_writer import write_csv
from rate_network import simulate
from rate_network_for_analysis import simulate as simulate_for_analysis

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=1, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--asp', metavar='asp', type=float, help='', default=0.)
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--frac_inputs_fixed', metavar='fi', type=float)
parser.add_argument('--syn_change_prob', metavar='cp', type=float, default=0.)
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--hd_hd_sparsity', metavar='dds', type=float, default=1.)
parser.add_argument('--hd_hr_sparsity', metavar='drs', type=float, default=1.)
parser.add_argument('--struct_prior', metavar='sp', type=str, default='shift')
parser.add_argument('--bump_init', metavar='bi', type=int, default=1)
parser.add_argument('--threshold_het', metavar='th', type=float, default=0)
parser.add_argument('--root_file_name', metavar='rfn', type=str, default=None)
parser.add_argument('--exp_title', metavar='et', type=str, default='')
parser.add_argument('--train', metavar='t', type=int, default=1)
parser.add_argument('--self_org_iters', metavar='sot', type=int, default=280)


args = parser.parse_args()
print(args)

np.random.seed(args.seed)

SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch if args.train else 1
self_org_iters = args.self_org_iters
decoder_train_trial_nums = (self_org_iters, self_org_iters + 20)
decoder_test_trial_nums = (self_org_iters + 20, self_org_iters + 120)
N_INNER_LOOP_RANGE = (self_org_iters + 120, self_org_iters + 121) # Number of times to simulate network and plasticity rules per loss function evaluation
READOUTS_PER_TRIAL = 40
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
TEST_REPEATS = 10
ROOT_FILE_NAME = args.root_file_name

T = 0.150 # Total duration of one network simulation
T_TEST = 0.150
dt = 1e-4 # Timestep
input_start = int(20e-3/dt)
input_end = int(100e-3/dt)
input_len = input_end - input_start
decoding_len = int(T / dt - input_start)
input_block_timesteps = int(INPUT_BLOCK_DURATION / dt)
t = np.linspace(0, T, int(T / dt))
n_e_pool = 15 # Number excitatory cells in sequence (also length of sequence)
n_e_side = 15
n_i = 1 # Number inhibitory cells
v_thresh_e = 0.1
v_thresh_i = 0
if args.train:
	train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
	test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
else:
	train_seeds = np.random.randint(0, 1e7, size=TEST_REPEATS)
	test_seeds = np.random.randint(0, 1e7, size=TEST_REPEATS)

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
out_dir = f'sims_out/{args.exp_title}_{args.struct_prior}_{BATCH_SIZE}_TH_{args.threshold_het}_STD_EXPL_{STD_EXPL}__L1_PENALTY_{joined_l1}_ACT_PEN_{args.asp}_CHANGEP_{CHANGE_PROB_PER_ITER}_SEED_{SEED}_{time_stamp}'
os.mkdir(out_dir)

# Make subdirectory for outputting CMAES info
os.mkdir(os.path.join(out_dir, 'outcmaes'))

# make subdirectories for weights, rates, etc.
if not args.train:
	weight_path = os.path.join(out_dir, 'weights')
	os.mkdir(weight_path)
	activity_path = os.path.join(out_dir, 'activities')
	os.mkdir(activity_path)
	integrated_value_path = os.path.join(out_dir, 'integrated_values')
	os.mkdir(integrated_value_path)
	inputs_path = os.path.join(out_dir, 'inputs')
	os.mkdir(inputs_path)

# Made CSVs for outputting train & test data
header = ['evals', 'loss'] + [f'true_loss_{i}' for i in np.arange(BATCH_SIZE)]
header += list(rule_names)
header += ['effect_means']
header += ['effect_stds']

train_data_path = os.path.join(out_dir, 'train_data.csv')
write_csv(train_data_path, header)

test_data_path = os.path.join(out_dir, 'test_data.csv')
write_csv(test_data_path, header)

# define weight values
if args.struct_prior == 'hard_coded':
	w_e_e = 0.6e-4 / dt
	w_pool_side = -0.2e-4 / dt
	w_side_pool = 0.6e-4 / dt
else:
	w_e_e = 9e-4 / dt * 0.1 / n_e_pool
	w_pool_side = -3e-4 / dt * 0.1 / n_e_pool
	w_side_pool = 9e-4 / dt * 0.1 / n_e_side

w_e_i = 2.5e-4 / dt / n_e_pool
w_i_e = -1e-4 / dt / n_i


def create_shuffled_one_to_one(size):
	w = np.diag(np.ones((size)))
	x = np.arange(size).astype(int)
	order = copy(x)
	np.random.shuffle(order)
	w[order, :] = w[np.arange(size), :]
	return w


def create_shift_matrix(size, k_start=1, k=1, ring=False):
	w = np.zeros((size, size))
	if k >= 1:
		for k_p in np.arange(k_start, k+1):
			w += np.diag(np.ones((size - k_p,)), k=k_p)
			### Add to make into a ring structure
			if ring:
				w[(size - k_p):, k - k_p] = 1

	elif k <= -1:
		for k_p in np.arange(np.abs(k_start), np.abs(k) + 1):
			w += np.diag(np.ones((size - k_p,)), k=-k_p)
			### Add to make into a ring structure
			if ring:
				w[-k - k_p, (size - k_p):] = 1
	return w


def make_hardcoded_network():
	w_initial = np.zeros((n_e_pool + 2 * n_e_side + n_i, n_e_pool + 2 * n_e_side + n_i))
	connectivity_scale = 0.075 * 15
	shift_mats = []

	for i in range(1, 10):
		w_shift = np.diag(np.ones(n_e_pool - np.abs(i)), k=i) * 1.5 * w_e_e * np.exp(-np.abs(i-1) / connectivity_scale)
		shift_mats.append(w_shift)
		shift_mats.append(np.transpose(w_shift))

	w_initial[:n_e_pool, :n_e_pool] = np.sum(np.stack(shift_mats), axis=0)

	# w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * np.random.rand(n_e_pool, n_e_side)
	# w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * np.random.rand(n_e_pool, n_e_side)

	w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * create_shift_matrix(n_e_side, k=3)
	w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * create_shift_matrix(n_e_side, k=-3)

	# w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = w_pool_side * np.random.rand(n_e_side, n_e_pool)
	# w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = w_pool_side * np.random.rand(n_e_side, n_e_pool)

	left_input_cells = w_pool_side * (1 - (create_shift_matrix(n_e_side, k=3) + create_shift_matrix(n_e_side, k=-3)))
	np.fill_diagonal(left_input_cells, 0)
	right_input_cells = copy(left_input_cells)

	w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = left_input_cells
	w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = right_input_cells

	w_initial[-n_i:, :n_e_pool] = gaussian_if_under_val(1, (n_i, n_e_pool), w_e_i, 0 * w_e_i)
	w_initial[:n_e_pool, -n_i:] = gaussian_if_under_val(1, (n_e_pool, n_i), w_i_e, 0 * np.abs(w_i_e))

	np.fill_diagonal(w_initial, 0)
	return w_initial


def make_network():
	'''
	Edit.

	'''
	if args.struct_prior == 'hard_coded':
		return make_hardcoded_network()

	w_initial = np.zeros((n_e_pool + 2 * n_e_side + n_i, n_e_pool + 2 * n_e_side + n_i))

	# sparsify e --> e connectivity to see in ring can be learned on top of heterogenous connectivity
	if args.struct_prior != 'seq':
		w_initial[:n_e_pool, :n_e_pool] = np.where(np.random.rand(n_e_pool, n_e_pool) < args.hd_hd_sparsity, w_e_e * np.random.rand(n_e_pool, n_e_pool), 0)
	else:
		w_initial[:n_e_pool, :n_e_pool] = w_e_e * np.random.rand(n_e_pool, n_e_pool) + np.diag(np.ones((n_e_pool - 1,)), k=-1)

	### For initializing a ring-like shape in the pool neurons

	# x = np.arange(n_e_pool) / n_e_pool
	# connectivity_scale = 0.075
	# exp_ring_connectivity = 4 * w_e_e * (np.exp(-x/connectivity_scale) + np.exp((x-1)/connectivity_scale))

	# for r_idx in np.arange(n_e_pool):
	# 	w_initial[r_idx:n_e_pool, r_idx] = exp_ring_connectivity[:(n_e_pool - r_idx)]
	# 	w_initial[0:r_idx, r_idx] = exp_ring_connectivity[(n_e_pool - r_idx):]

	# w_initial[:n_e_pool, :n_e_pool] = w_initial[:n_e_pool, :n_e_pool] * np.random.normal(size=(n_e_pool, n_e_pool), loc=1, scale=0.1)

	###
	
	if args.struct_prior == 'shift' or args.struct_prior == 'ring':
		init_ring = (args.struct_prior == 'ring')
		# define connectivity from HR to HD neurons as "shift" matrix
		w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * np.where(np.random.rand(n_e_pool, n_e_side) < args.hd_hr_sparsity, create_shift_matrix(n_e_side, k=3, ring=init_ring), 0)
		w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool *  np.where(np.random.rand(n_e_pool, n_e_side) < args.hd_hr_sparsity, create_shift_matrix(n_e_side, k=-3, ring=init_ring), 0)

		# define connectivity from HD to HR as inhibiting all but the corresponding group along the diagonal
		left_input_cells = w_pool_side * (1 - (create_shift_matrix(n_e_side, k=3, ring=init_ring) + create_shift_matrix(n_e_side, k=-3, ring=init_ring)))
		np.fill_diagonal(left_input_cells, 0)
		right_input_cells = copy(left_input_cells)

		w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = np.where(np.random.rand(n_e_side, n_e_pool) < args.hd_hr_sparsity, left_input_cells, 0)
		w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = np.where(np.random.rand(n_e_side, n_e_pool) < args.hd_hr_sparsity, right_input_cells, 0)
	elif args.struct_prior == 'one_in_one_out':
		# define connectivity from HR to HD neurons as random, semi-sparse matrix
		w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * (create_shuffled_one_to_one(n_e_side) + np.random.rand(n_e_pool, n_e_side) * 0.05)
		w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * (create_shuffled_one_to_one(n_e_side) + np.random.rand(n_e_pool, n_e_side) * 0.05)

		# define connectivity from HD to HR neurons as random, semi-sparse matrix
		w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = w_pool_side * (1 - create_shuffled_one_to_one(n_e_side) + np.random.rand(n_e_side, n_e_pool) * 0.05)
		w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = w_pool_side * (1 - create_shuffled_one_to_one(n_e_side) + np.random.rand(n_e_side, n_e_pool) * 0.05)
	elif args.struct_prior == 'random' or args.struct_prior == 'seq':
		# define connectivity from HR to HD neurons as random, semi-sparse matrix
		w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * np.where(np.random.rand(n_e_pool, n_e_side) < args.hd_hr_sparsity, np.random.rand(n_e_pool, n_e_side), 0)
		w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * np.where(np.random.rand(n_e_pool, n_e_side) < args.hd_hr_sparsity, np.random.rand(n_e_pool, n_e_side), 0)

		# define connectivity from HD to HR neurons as random, semi-sparse matrix
		w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = w_pool_side * np.where(np.random.rand(n_e_side, n_e_pool) < args.hd_hr_sparsity, np.random.rand(n_e_side, n_e_pool), 0)
		w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = w_pool_side * np.where(np.random.rand(n_e_side, n_e_pool) < args.hd_hr_sparsity, np.random.rand(n_e_side, n_e_pool), 0)
		

	w_initial[-n_i:, :n_e_pool] = gaussian_if_under_val(1, (n_i, n_e_pool), w_e_i, 0 * w_e_i)
	w_initial[:n_e_pool, -n_i:] = gaussian_if_under_val(1, (n_e_pool, n_i), w_i_e, 0 * np.abs(w_i_e))

	np.fill_diagonal(w_initial, 0)
	return w_initial


def calc_loss(r : np.ndarray, train_diff_drives : np.ndarray, test_diff_drives : np.ndarray, readout_times : np.ndarray):

	if np.isnan(r).any():
		return 10000

	r_readout = r[:, :, :n_e_pool]

	stacked_activities_train = []
	y_train = []
	stacked_activities_test = []
	y_test = []

	for i in range(readout_times.shape[0]):
		trial_num = int(i / READOUTS_PER_TRIAL)
		if i < train_diff_drives.shape[0] * READOUTS_PER_TRIAL:
			stacked_activities_train.append(r_readout[trial_num, readout_times[i], :].flatten())
			y_train.append(train_diff_drives[trial_num, readout_times[i] - input_start])
		else:
			stacked_activities_test.append(r_readout[trial_num, readout_times[i], :].flatten())
			y_test.append(test_diff_drives[trial_num - train_diff_drives.shape[0], readout_times[i] - input_start])

	X_train = np.stack(stacked_activities_train)
	y_train = np.array(y_train)

	X_test = np.stack(stacked_activities_test)
	y_test = np.array(y_test)

	reg = LinearRegression().fit(X_train, y_train)
	loss = 1000 * (1 - reg.score(X_test, y_test))

	return loss


def plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, total_activity_penalties, train=True):
	scale = 3
	n_res_to_show = BATCH_SIZE

	gs = gridspec.GridSpec(4 * n_res_to_show + 3, 2)
	fig = plt.figure(figsize=(4  * scale, (4 * n_res_to_show + 3) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(4 * n_res_to_show)]
	axs += [fig.add_subplot(gs[4 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[4 * n_res_to_show + 1, :])]
	axs += [fig.add_subplot(gs[4 * n_res_to_show + 2, :])]

	all_effects = []

	for i in np.arange(BATCH_SIZE):
		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
		res = results[i]
		r = res['r']
		r_exp_filtered = res['r_exp_filtered']
		w = res['w']
		w_initial = res['w_initial']
		effects = res['syn_effects']
		all_weight_deltas = res['all_weight_deltas']
		rs_for_loss = res['rs_for_loss']
		r_in_for_loss = res['r_in_for_loss']
		targets_for_loss = res['targets_for_loss']

		all_effects.append(effects)

		plotted_trial_count = 0

		for trial_idx in range(rs_for_loss.shape[0]):
			if trial_idx < rs_for_loss.shape[0] - 3:
				continue
			r = rs_for_loss[trial_idx, ...]
			r_in = r_in_for_loss[trial_idx, ...]
			targets = targets_for_loss[trial_idx, ...]

			# scale = 1
			# fig_r_in, axs_r_in = plt.subplots(1, 1, figsize=(4 * scale, 2 * scale), sharex=True, sharey=True)
			# axs_r_in.plot(np.arange(len(targets)), targets, color='red')
			# input_diffs = r_in[input_start:, n_e_pool + n_e_side : n_e_pool + 2 * n_e_side].sum(axis=1) - r_in[input_start:, n_e_pool:n_e_pool + n_e_side].sum(axis=1)
			# input_summed = [input_diffs[:j].sum() for j in range(len(input_diffs))]
			# axs_r_in.plot(np.arange(len(targets)), input_summed, color='black')

			# fig_r_in.savefig(f'{out_dir}/r_in_trial_{trial_idx}.png')

			for l_idx in range(r.shape[1]):
				if l_idx < n_e_pool:
					pass
					# if l_idx % 1 == 0:
					# 	axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
				elif l_idx >= (r.shape[1] - n_i):
					axs[4 * i + plotted_trial_count][1].plot(np.arange(len(r[:, l_idx])), r[:, l_idx], c='black') # graph inh activity

			axs[4 * i + plotted_trial_count][0].matshow(r[:, :n_e_pool + 2 * n_e_side].T, aspect=1/0.1)
			plotted_trial_count += 1

		vbound = np.max(w)

		mappable = axs[4 * i + 3][0].matshow(w_initial, vmin=-vbound, vmax=vbound, cmap='bwr') # plot initial weight matrix
		plt.colorbar(mappable, ax=axs[4 * i + 3][0])

		mappable = axs[4 * i + 3][1].matshow(w, vmin=-vbound, vmax=vbound, cmap='bwr') # plot final weight matrix
		plt.colorbar(mappable, ax=axs[4 * i + 3][1])

		axs[4 * i][0].set_title(f'{true_losses[i]} + {syn_effect_penalties[i]} + {total_activity_penalties[i]}')
		for i_axs in range(2):
			axs[2 * i][i_axs].set_xlabel('Time (s)')
			axs[2 * i][i_axs].set_ylabel('Firing rate')

		axs[4 * n_res_to_show + 2].plot(np.arange(len(all_weight_deltas)), np.log(all_weight_deltas), label=f'{i}')

	partial_rules_len = int(len(plasticity_coefs))

	all_effects = np.array(all_effects)
	effects = np.mean(all_effects, axis=0)

	axs[4 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
	effects_argsort = []
	for l in range(1):
		effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
		effects_argsort_partial = np.flip(np.argsort(effects_partial))
		effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
		x = np.arange(len(effects_argsort_partial)) + l * partial_rules_len
		axs[4 * n_res_to_show + 1].bar(x, effects_partial[effects_argsort_partial], zorder=0)
		for i_e in x:
			axs[4 * n_res_to_show + 1].scatter(i_e * np.ones(all_effects.shape[0]), all_effects[:, effects_argsort_partial][:, i_e], c='black', zorder=1, s=3)
	axs[4 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
	axs[4 * n_res_to_show + 1].set_xlim(-1, len(effects))

	true_loss = np.sum(true_losses)
	syn_effect_penalty = np.sum(syn_effect_penalties)
	total_activity_penalty = np.sum(total_activity_penalties)
	axs[4 * n_res_to_show].set_title(f'Loss: {true_loss + syn_effect_penalty}, {true_loss}, {syn_effect_penalty}, {total_activity_penalty}')

	# plot the coefficients assigned to each plasticity rule (unsorted by size)
	for l in range(1):
		axs[4 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
	axs[4 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
	axs[4 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
	axs[4 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))

	axs[4 * n_res_to_show + 2].set_xlabel('Epochs')
	axs[4 * n_res_to_show + 2].set_ylabel('log(delta W)')
	axs[4 * n_res_to_show + 2].legend()

	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	# fig.tight_layout()
	if train:
		fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	else:
		fig.savefig(f'{out_dir}/{zero_padding}{evals}_test.png')
	plt.close('all')


def calc_alpha_func(tau_alpha):
	alpha_func_n_steps = int(10 * tau_alpha / dt)
	t_alpha = np.arange(0, alpha_func_n_steps) * dt
	return np.e * t_alpha / tau_alpha * np.exp(-t_alpha/tau_alpha)


def poisson_arrivals_to_inputs(arrivals, tau_alpha):
	alpha_func = calc_alpha_func(tau_alpha)
	input_current = np.zeros(arrivals.shape)

	for i in range(arrivals.shape[1]):
		input_current[:, i] = np.convolve(alpha_func, arrivals[:, i], mode='full')[:arrivals.shape[0]]
	return input_current


def simulate_single_network(index, x, train, track_params=True):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	plasticity_coefs = x[:N_RULES]
	rule_time_constants = x[N_RULES:]

	t = np.linspace(0, T, int(T / dt))

	if FIXED_DATA:
		if train:
			print(train_seeds[index])
			sys.stdout.flush()
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network() # make a new ring attractor
	v_e = np.random.normal(loc=v_thresh_e, scale=args.threshold_het, size=(n_e_pool + 2 * n_e_side,))
	v_e = np.where(v_e > 0, v_e, 0)
	v_thresh = np.concatenate([
		v_e,
		v_thresh_i * np.ones((n_i,)),
	])

	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	num_readouts = (decoder_train_trial_nums[1] - decoder_train_trial_nums[0] + decoder_test_trial_nums[1] - decoder_test_trial_nums[0]) * READOUTS_PER_TRIAL
	readout_times = (np.random.rand(num_readouts) * decoding_len + input_start).astype(int)

	input_signal_totals = np.zeros((n_inner_loop_iters, decoding_len))

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	all_effects = np.zeros(plasticity_coefs.shape)
	normed_loss = 10000	
	rs_for_loss = []
	r_in_for_loss = []
	targets_for_loss = []

	w_hist = []
	all_weight_deltas = []
	w_hist.append(w)

	blew_up = False

	surviving_synapse_mask = np.ones((n_e_pool, n_e_pool)).astype(bool)

	for i in range(n_inner_loop_iters):
		# print(f'Activation number: {i}')
		# Define input for activation of the network

		if i == decoder_train_trial_nums[0]:
			t = np.linspace(0, T_TEST, int(T_TEST / dt))

		input_spks = np.zeros((decoding_len, 2 * n_e_side))
		inputs = np.zeros((decoding_len,)).astype(int)
		inputs[0] = 1

		for k in range(input_len):
			if k % input_block_timesteps == 0:
				inputs[k] = np.random.choice([-1, 0, 1])
				if inputs[k] == -1:
					input_block = np.random.poisson(lam=2 * INPUT_RATE_PER_CELL * dt, size=(input_block_timesteps, n_e_side))
					input_spks[k : k + input_block_timesteps, :n_e_side] = input_block
				elif inputs[k] == 1:
					input_block = np.random.poisson(lam=2 * INPUT_RATE_PER_CELL * dt, size=(input_block_timesteps, n_e_side))
					input_spks[k : k + input_block_timesteps, n_e_side : 2 * n_e_side] = input_block
			else:
				inputs[k] = inputs[k-1]

		filtered_input_to_sum_per_neuron = poisson_arrivals_to_inputs(input_spks, 3e-3)
		filtered_input_to_sum = filtered_input_to_sum_per_neuron[:, n_e_side:2 * n_e_side].sum(axis=1) - filtered_input_to_sum_per_neuron[:, :n_e_side].sum(axis=1)
		running_input_sums = np.zeros_like(filtered_input_to_sum)
		for j in range(len(running_input_sums)):
			if j > 0:
				running_input_sums[j] += running_input_sums[j-1]
			running_input_sums[j] += filtered_input_to_sum[j]

		r_in_spks = np.zeros((len(t), n_e_pool + 2 * n_e_side + n_i))
		input_size = 6
		input_slice = slice(int((n_e_pool - input_size)/ 2), int((n_e_pool + input_size)/ 2))
		if bool(args.bump_init):
			r_in_spks[:int(10e-3/dt), input_slice] = np.random.poisson(lam=INPUT_RATE_PER_CELL * dt, size=(int(10e-3/dt), 6))

		r_in_spks[input_start:decoding_len + input_start, n_e_pool:n_e_pool + 2 * n_e_side] = input_spks
		r_in = poisson_arrivals_to_inputs(r_in_spks, 3e-3)
		
		input_signal_totals[i, :] = running_input_sums / input_len

		r_in[:, :n_e_pool]  = 0.25 * r_in[:, :n_e_pool]
		r_in[:, n_e_pool:(n_e_pool + 2 * n_e_side)] = 0.1 * r_in[:, n_e_pool:(n_e_pool + 2 * n_e_side)]

		r_in[:, :n_e_pool] += 0.02 * poisson_arrivals_to_inputs(np.random.poisson(lam=INPUT_RATE_PER_CELL * dt, size=(len(t), n_e_pool)), 3e-3)

		# if i <= 400:
		# 	synapse_change_mask_for_i = np.random.rand(n_e, n_e) < CHANGE_PROB_PER_ITER

		# 	drop_mask_for_i = np.logical_and(synapse_change_mask_for_i, surviving_synapse_mask)
		# 	birth_mask_for_i = np.logical_and(synapse_change_mask_for_i, ~surviving_synapse_mask)

		# 	surviving_synapse_mask[synapse_change_mask_for_i] = ~surviving_synapse_mask[synapse_change_mask_for_i]

		# 	w[:n_e, :n_e] = np.where(drop_mask_for_i, 0, w[:n_e, :n_e])
		# 	w[:n_e, :n_e] = np.where(birth_mask_for_i, w_e_e_added, w[:n_e, :n_e])

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects, r_exp_filtered = simulate(t, n_e_pool, n_e_side, n_i, r_in, plasticity_coefs, rule_time_constants, w, w_plastic, v_thresh, dt=dt, tau_e=5e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

		if not args.train:
			# save weights
			weight_file_name = os.path.join(weight_path, f'net_{zero_pad(index, 3)}_act_{zero_pad(i, 4)}.npy')
			np.save(weight_file_name, w)
			# save activity
			activity_file_name = os.path.join(activity_path, f'net_{zero_pad(index, 3)}_act_{zero_pad(i, 4)}.npy')
			np.save(activity_file_name, r)
			# save integrated_values
			integrated_value_file_name = os.path.join(integrated_value_path, f'net_{zero_pad(index, 3)}_act_{zero_pad(i, 4)}.npy')
			np.save(integrated_value_file_name, input_signal_totals[i, :])
			# save inputs
			inputs_file_name = os.path.join(inputs_path, f'net_{zero_pad(index, 3)}_act_{zero_pad(i, 4)}.npy')
			np.save(inputs_file_name, filtered_input_to_sum_per_neuron)

		if (np.isnan(r).any()
	  		or (np.abs(w_out) > 100).any()
			or (np.abs(w_out[:n_e_pool, :n_e_pool]) < 1.5e-6).all() 
			or (np.abs(w_out[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)]) < 1.5e-6).all()
			or (np.abs(w_out[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool]) < 1.5e-6).all()): # if simulation turns up nans in firing rate matrix, end the simulation
			
			return {
				'blew_up': True,
			}
			
		if (i >= decoder_train_trial_nums[0] and i < decoder_train_trial_nums[1]) or (i >= decoder_test_trial_nums[0] and i < decoder_test_trial_nums[1]):
			rs_for_loss.append(r)
			r_in_for_loss.append(r_in)
			targets_for_loss.append(running_input_sums)

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects[:N_RULES]

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	train_diffs = input_signal_totals[decoder_train_trial_nums[0]:decoder_train_trial_nums[1], :]
	test_diffs = input_signal_totals[decoder_test_trial_nums[0]:decoder_test_trial_nums[1], :]

	rs_for_loss = np.stack(rs_for_loss)
	normed_loss = calc_loss(rs_for_loss, train_diffs, test_diffs, readout_times)

	return {
		'loss': normed_loss,
		'blew_up': False,
		'r': r,
		'rs_for_loss': rs_for_loss,
		'r_in_for_loss': np.stack(r_in_for_loss),
		'targets_for_loss': np.stack(targets_for_loss),
		'r_exp_filtered': r_exp_filtered,
		'w': w,
		'w_initial': w_initial,
		'syn_effects': all_effects,
		'all_weight_deltas': all_weight_deltas,
	}


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
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss'] or not args.train:
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


def simulate_single_network_wrapper(tup):
	return simulate_single_network(*tup)


def eval_all(X, eval_tracker=None, train=True):
	start = time.time()

	indices = np.arange(BATCH_SIZE)
	pool = mp.Pool(POOL_SIZE)

	task_vars = []
	
	if args.train:
		for x in X:
			for idx in indices:
				task_vars.append((idx, x, train))
	else:
		for i_x, x in enumerate(X):
			for idx in indices:
				task_vars.append((i_x, x, train))
	results = pool.map(simulate_single_network_wrapper, task_vars)

	pool.close()
	pool.join()

	losses = []
	for i in range(len(X)):
		loss, true_losses, syn_effects, syn_effect_penalties, total_activities = process_plasticity_rule_results(results[BATCH_SIZE * i: BATCH_SIZE * (i+1)], X[i], eval_tracker=eval_tracker, train=train)
		losses.append(loss)
		if train:
			log_sim_results(train_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
		else:
			log_sim_results(test_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
	
	dur = time.time() - start
	print('dur:', dur)
	sys.stdout.flush()

	return losses


def process_params_str(s):
	params = []
	for x in s.split(' '):
		x = x.replace('\n', '')
		if x != '':
			params.append(float(x))
	return np.array(params)


def load_best_avg_params(file_names, n_plasticity_coefs, n_time_constants, batch_size):
	all_best_syn_effects = []
	all_best_coefs = []

	for file_name in file_names:
		test_data_path = f'./sims_out/{file_name}/test_data.csv'
		df_test = read_csv(test_data_path, read_header=False)

		syn_effect_start = 2 + batch_size + n_plasticity_coefs + n_time_constants
		syn_effect_end = 2 + batch_size + n_plasticity_coefs + n_time_constants + n_plasticity_coefs
		plasticity_coefs_start = 2 + batch_size
		plasticity_coefs_end = 2 + batch_size + n_plasticity_coefs + n_time_constants

		x = np.arange(df_test.shape[0])
		losses_test = df_test[df_test.columns[1]]
		x_best_min_test = np.argmin(losses_test)

		print('loading plasticity with best loss:', losses_test[x_best_min_test])

		final_syn_effects = []
		for i in range(syn_effect_start, syn_effect_end):
			final_syn_effects.append(df_test[df_test.columns[i]][x_best_min_test])
		final_syn_effects = np.array(final_syn_effects)

		final_coefs = []
		for i in range(plasticity_coefs_start, plasticity_coefs_end):
			final_coefs.append(df_test[df_test.columns[i]][x_best_min_test])
		final_coefs = np.array(final_coefs)

		all_best_syn_effects.append(final_syn_effects)
		all_best_coefs.append(final_coefs)

	return np.mean(np.stack(all_best_syn_effects), axis=0), np.mean(np.stack(all_best_coefs), axis=0)


if __name__ == '__main__':
	mp.set_start_method('fork')

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
	}

	if not args.train:
		if args.struct_prior == 'hard_coded':
			x_test = x0 = np.concatenate([np.zeros(N_RULES), 5e-3 * np.ones(N_TIMECONSTS)])
		else:
			# Load learned synaptic rules from root_file_name
			file_names = [ROOT_FILE_NAME]
			syn_effects_test, x_test = load_best_avg_params(file_names, N_RULES, N_TIMECONSTS, 10)
			print(x_test)

		eval_all([x_test] * TEST_REPEATS, eval_tracker=eval_tracker)

	else:

		if args.load_initial is not None:
			x0 = load_best_params(args.load_initial)
		else:
			x0 = np.concatenate([np.zeros(N_RULES), 5e-3 * np.ones(N_TIMECONSTS)])


		eval_all([x0], eval_tracker=eval_tracker)

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

		while not es.stop():
			X = es.ask()
			print(X)
			es.tell(X, eval_all(X, eval_tracker=eval_tracker))
			if eval_tracker['best_changed']:
				eval_all([eval_tracker['params']], eval_tracker=eval_tracker, train=False)
			es.disp()
