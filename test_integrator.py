from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val, rev_argsort, set_smallest_n_zero
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
from numba import njit
from scipy.sparse import csc_matrix
from sklearn.linear_model import LinearRegression
from csv_reader import read_csv
from csv_writer import write_csv

from rate_network import simulate, tanh, generate_gaussian_pulse

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
parser.add_argument('--root_file_name', metavar='rfn', type=str)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_RANGE = (320, 321) # Number of times to simulate network and plasticity rules per loss function evaluation
decoder_train_trial_nums = (280, 300)
decoder_test_trial_nums = (300, 320)
STD_EXPL = args.std_expl
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_LOSS_COEF = args.asp
CHANGE_PROB_PER_ITER = args.syn_change_prob #0.0007
FRAC_INPUTS_FIXED = args.frac_inputs_fixed
INPUT_RATE_PER_CELL = 1000
N_RULES = 60 + 8
N_TIMECONSTS = 36 + 16
REPEATS = 5
ROOT_FILE_NAME = args.root_file_name

T = 0.12 # Total duration of one network simulation
dt = 1e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
n_e_pool = 15 # Number excitatory cells in sequence (also length of sequence)
n_e_side = 15
n_i = 1 # Number inhibitory cells
train_seeds = np.random.randint(0, 1e7, size=REPEATS)
test_seeds = np.random.randint(0, 1e7, size=REPEATS)

layer_colors = get_ordered_colors('gist_rainbow', 15)
np.random.shuffle(layer_colors)

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
	r'$\tilde{x} y, z~y$',
	r'$x \tilde{y}, z~y$',

	r'$w \tilde{x} y, z~y$',
	r'$w x \tilde{y}, z~y$',
]

rule_names += [
	[r'HD \rightarrow HD, HR~HD' + r_name for r_name in rule_names_tripartite],
	[r'HR \rightarrow HD, HD~HD' + r_name for r_name in rule_names_tripartite],
]


rule_names = [r for rs in rule_names for r in rs]
rule_names = np.array(rule_names, dtype=object)


# Make directory for outputting simulations
if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
joined_l1 = '_'.join([str(p) for p in L1_PENALTIES])
out_dir = f'sims_out/ring_int_test_rand{BATCH_SIZE}_STD_EXPL_{STD_EXPL}_FIXED_{FIXED_DATA}_L1_PENALTY_{joined_l1}_ACT_PEN_{args.asp}_CHANGEP_{CHANGE_PROB_PER_ITER}_FRACI_{FRAC_INPUTS_FIXED}_SEED_{SEED}_{time_stamp}'
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


w_e_e = 0.6e-4 / dt * 0.1
w_pool_side = -0.2e-4 / dt * 0.1
w_side_pool = 0.3e-4 / dt * 0.1

w_e_i = 2.5e-4 / dt / n_e_pool
w_i_e = -1e-4 / dt / n_i

w_e_e_added = 0.05 * w_e_e * 0.2

def create_shift_matrix(size, k=1):
	w = np.zeros((size, size))
	if k >= 1:
		for k_p in np.arange(1, k+1):
			w += np.diag(np.ones((size - k_p,)), k=k_p)
			w[(size - k_p):, k - k_p] = 1

	elif k <= -1:
		for k_p in np.arange(1, -k+1):
			w += np.diag(np.ones((size - k_p,)), k=-k_p)
			w[-k - k_p, (size - k_p):] = 1
	return w

def make_network():
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((n_e_pool + 2 * n_e_side + n_i, n_e_pool + 2 * n_e_side + n_i))

	w_initial[:n_e_pool, :n_e_pool] = w_e_e * np.random.rand(n_e_pool, n_e_pool)

	# x = np.arange(n_e_pool) / n_e_pool
	# connectivity_scale = 0.075
	# exp_ring_connectivity = 4 * w_e_e * (np.exp(-x/connectivity_scale) + np.exp((x-1)/connectivity_scale))

	# for r_idx in np.arange(n_e_pool):
	# 	w_initial[r_idx:n_e_pool, r_idx] = exp_ring_connectivity[:(n_e_pool - r_idx)]
	# 	w_initial[0:r_idx, r_idx] = exp_ring_connectivity[(n_e_pool - r_idx):]
	
	w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * np.random.rand(n_e_pool, n_e_side)
	# w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = np.where(np.random.rand(n_e_pool, n_e_side) >= 0.5, w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)], 0)
	w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * np.random.rand(n_e_pool, n_e_side)
	# w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = np.where(np.random.rand(n_e_pool, n_e_side) >= 0.5, w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)], 0)

	# w_initial[:n_e_pool, n_e_pool:(n_e_pool + n_e_side)] = w_side_pool * create_shift_matrix(n_e_side, k=3)
	# w_initial[:n_e_pool, (n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side)] = w_side_pool * create_shift_matrix(n_e_side, k=-3)

	w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = w_pool_side * np.random.rand(n_e_side, n_e_pool)
	w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = w_pool_side * np.random.rand(n_e_side, n_e_pool)

	# left_input_cells = w_pool_side * (1 - (create_shift_matrix(n_e_side, k=3) + create_shift_matrix(n_e_side, k=-3)))
	# np.fill_diagonal(left_input_cells, 0)
	# right_input_cells = copy(left_input_cells)

	# w_initial[n_e_pool:(n_e_pool + n_e_side), :n_e_pool] = left_input_cells
	# w_initial[(n_e_pool + n_e_side):(n_e_pool + 2 * n_e_side), :n_e_pool] = right_input_cells

	w_initial[-n_i:, :n_e_pool] = gaussian_if_under_val(1, (n_i, n_e_pool), w_e_i, 0 * w_e_i)
	w_initial[:n_e_pool, -n_i:] = gaussian_if_under_val(1, (n_e_pool, n_i), w_i_e, 0 * np.abs(w_i_e))

	np.fill_diagonal(w_initial, 0)
	return w_initial


def calc_loss(r : np.ndarray, train_diff_drives : np.ndarray, test_diff_drives : np.ndarray, readout_times : np.ndarray):

	if np.isnan(r).any():
		return 10000

	r_readout = r[:, :, :n_e_pool]

	stacked_activities_train = []
	stacked_activities_test = []

	for i in range(r.shape[0]):
		if i < train_diff_drives.shape[0]:
			stacked_activities_train.append(r_readout[i, readout_times[i], :].flatten())
		else:
			stacked_activities_test.append(r_readout[i, readout_times[i], :].flatten())

	X_train = np.stack(stacked_activities_train)
	y_train = train_diff_drives

	print(X_train)
	print(y_train)

	X_test = np.stack(stacked_activities_test)
	y_test = test_diff_drives

	reg = LinearRegression().fit(X_train, y_train)

	print(reg.coef_)

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

		all_effects.append(effects)

		plotted_trial_count = 0

		for trial_idx in range(rs_for_loss.shape[0]):
			if trial_idx < rs_for_loss.shape[0] - 3:
				continue
			r = rs_for_loss[trial_idx, ...]

			for l_idx in range(r.shape[1]):
				if l_idx < n_e_pool:
					pass
					# if l_idx % 1 == 0:
					# 	axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
				elif l_idx >= (r.shape[1] - n_i):
					axs[4 * i + plotted_trial_count][1].plot(t, r[:, l_idx], c='black') # graph inh activity

			axs[4 * i + plotted_trial_count][0].matshow(r[:, :n_e_pool + 2 * n_e_side].T, aspect=1/0.1)
			plotted_trial_count += 1

		r_exc = r[:, :n_e_pool]
		r_summed = np.sum(r_exc, axis=0)
		r_active_mask =  np.where(r_summed != 0, 1, 0).astype(bool)
		r_summed_safe_divide = np.where(r_active_mask, r_summed, 1)
		r_normed = r_exc / r_summed_safe_divide
		t_means = np.sum(t.reshape(t.shape[0], 1) * r_normed, axis=0)
		# t_ordering = np.argsort(t_means)
		# t_ordering = np.concatenate([t_ordering, np.arange(n_e, n_e + n_i)])

		# sorted_w_initial = w_initial[t_ordering, :][:, t_ordering]
		# sorted_w = w[t_ordering, :][:, t_ordering]

		vmin = np.min([w_initial.min(), w.min()])
		vmax = np.max([w_initial.max(), w.max()])

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

	fig.tight_layout()
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

	if FIXED_DATA:
		if train:
			print(train_seeds[index])
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network() # make a new ring attractor

	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	input_start = int(20e-3/dt)
	input_end = int(100e-3/dt)
	input_len = input_end - input_start

	num_readouts = decoder_train_trial_nums[1] - decoder_train_trial_nums[0] + decoder_test_trial_nums[1] - decoder_test_trial_nums[0]
	readout_times = ((input_end * dt + 15e-3 * (1 - np.sqrt(1 - np.random.rand(num_readouts)))) / dt).astype(int)

	input_signal_transition_probs = np.random.rand(n_inner_loop_iters, 2) * 0.05 + 0.95
	input_signal_totals = np.zeros((n_inner_loop_iters,))

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	all_effects = np.zeros(plasticity_coefs.shape)
	normed_loss = 10000	
	rs_for_loss = []

	w_hist = []
	all_weight_deltas = []
	w_hist.append(w)

	blew_up = False

	surviving_synapse_mask = np.ones((n_e_pool, n_e_pool)).astype(bool)

	for i in range(n_inner_loop_iters):
		# print(f'Activation number: {i}')
		# Define input for activation of the network
		r_in_spks = np.zeros((len(t), n_e_pool + 2 * n_e_side + n_i))
		r_in_spks[:int(15e-3/dt), :6] = np.random.poisson(lam=INPUT_RATE_PER_CELL * dt, size=(int(15e-3/dt), 6))

		input_spks = np.random.poisson(lam=2 * INPUT_RATE_PER_CELL * dt, size=(input_len, n_e_side))
		input_signal_transition_probs_i = input_signal_transition_probs[i, :]
		markov_state = np.zeros((input_len,)).astype(int)
		markov_state[0] = 1
		for k in range(input_len - 1):
			if input_signal_transition_probs_i[markov_state[k]] > np.random.rand():
				markov_state[k+1] = markov_state[k]
			else:
				markov_state[k+1] = 1 - markov_state[k]

		input_signal_totals[i] = np.mean(markov_state)

		# right_input_spks = np.logical_and(input_spks, rnd_walk_steps_i > 0)
		input_spks[np.nonzero(1 - markov_state)[0], :] = 0

		# print('input spikes', np.sum(input_spks))
		# print('input diffs', input_signal_totals[i])

		# r_in_spks[input_start:input_end, n_e_pool:n_e_pool + n_e_side] = right_input_spks
		r_in_spks[input_start:input_end, n_e_pool + n_e_side:n_e_pool + 2 * n_e_side] = input_spks
		r_in = poisson_arrivals_to_inputs(r_in_spks, 3e-3)

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
		r, s, v, w_out, effects, r_exp_filtered = simulate(t, n_e_pool, n_e_side, n_i, r_in, plasticity_coefs, rule_time_constants, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

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

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects[:N_RULES]

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	train_diffs = input_signal_totals[decoder_train_trial_nums[0]:decoder_train_trial_nums[1]]
	test_diffs = input_signal_totals[decoder_test_trial_nums[0]:decoder_test_trial_nums[1]]

	rs_for_loss = np.stack(rs_for_loss)
	normed_loss = calc_loss(rs_for_loss, train_diffs, test_diffs, readout_times)

	return {
		'loss': normed_loss,
		'blew_up': False,
		'r': r,
		'rs_for_loss': rs_for_loss,
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


def simulate_single_network_wrapper(tup):
	return simulate_single_network(*tup)


def eval_all(X, eval_tracker=None, train=True):
	start = time.time()

	indices = np.arange(BATCH_SIZE)
	pool = mp.Pool(POOL_SIZE)

	task_vars = []
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

	# Load learned synaptic rules from root_file_name
	file_names = [ROOT_FILE_NAME]

	syn_effects_test, x_test = load_best_avg_params(file_names, N_RULES, N_TIMECONSTS, 5)
	print(x_test)

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
	}

	eval_all([x_test] * REPEATS, eval_tracker=eval_tracker)
