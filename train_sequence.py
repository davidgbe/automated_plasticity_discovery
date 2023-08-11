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
from cmaes_test_tools import generate_surrogate_problem

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, nargs=3, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=3, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--asp', metavar='asp', type=int, help="Flag for penalizing shape of neurons' activity ")
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--load_initial', metavar='li', type=str, help='File from which to load the best params as an initial guess')
parser.add_argument('--frac_inputs_fixed', metavar='fi', type=float)
parser.add_argument('--syn_change_prob', metavar='cp', type=float, default=0.)
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--tol_func_hist', metavar='tfh', type=float, default=1e-12)
parser.add_argument('--tol_func', metavar='tf', type=float, default=1e-11)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_RANGE = (399, 400) # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL_RULES = args.std_expl[0]
STD_EXPL_CONSTS= args.std_expl[1]
STD_EXPL_INH= args.std_expl[2]
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_LOSS_COEF = 6 if bool(args.asp) else 0
ACTIVITY_JITTER_COEF = 60
CHANGE_PROB_PER_ITER = args.syn_change_prob #0.0007
FRAC_INPUTS_FIXED = args.frac_inputs_fixed
INPUT_RATE_PER_CELL = 80
N_RULES = 34
N_TIMECONSTS = 16
N_RESTARTS = 10

T = 0.13 # Total duration of one network simulation
dt = 2e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
n_e = 20 # Number excitatory cells in sequence (also length of sequence)
n_i = 8 # Number inhibitory cells
train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)

layer_colors = get_ordered_colors('gist_rainbow', 15)
np.random.shuffle(layer_colors)

rule_names = [ # Define labels for all rules to be run during simulations
	r'',
	r'$y$',
	r'$x$',
	r'$x \, y$',
	r'$\sigma(y)$',
	r'$\sigma(x)$',
	r'$x \sigma(y)$',
	r'$\sigma(x) y $',
	r'$\sigma(x) \sigma(y) $',

	r'$x \, \tilde{y}$',
	r'$\tilde{x} \, y$',
	r'$\sigma(x) \, \tilde{y}$',
	r'$\tilde{x} \, \sigma(y)$',
	r'$\tilde{y} \, y$',
	r'$\tilde{x} \, x$',
	r'$\tilde{y}^2$',
	r'$\tilde{x}^2$',

	r'w',
	r'$w y$',
	r'$w x$',
	r'$w x \, y$',
	r'$w \sigma(y)$',
	r'$w \sigma(x)$',
	r'$w x \sigma(y)$',
	r'$w \sigma(x) y $',
	r'$w \sigma(x) \sigma(y) $',

	r'$w x \, \tilde{y}$',
	r'$w \tilde{x} \, y$',
	r'$w \sigma(x) \, \tilde{y}$',
	r'$w \tilde{x} \, \sigma(y)$',
	r'$w \tilde{y} \, y$',
	r'$w \tilde{x} \, x$',
	r'$w \tilde{y}^2$',
	r'$w \tilde{x}^2$',
]

rule_names = [
	[r'$E \rightarrow E$ ' + r_name for r_name in rule_names],
	# [r'$E \rightarrow I$ ' + r_name for r_name in rule_names],
	# [r'$I \rightarrow E$ ' + r_name for r_name in rule_names],
]
rule_names = np.array(rule_names).flatten()


# Make directory for outputting simulations
if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
joined_l1 = '_'.join([str(p) for p in L1_PENALTIES])
out_dir = f'sims_out/decoder_ee_3_extended_bipop_d_{BATCH_SIZE}_STD_EXPL_{STD_EXPL_RULES}_{STD_EXPL_CONSTS}_{STD_EXPL_INH}_FIXED_{FIXED_DATA}_L1_PENALTY_{joined_l1}_ACT_PEN_{args.asp}_CHANGEP_{CHANGE_PROB_PER_ITER}_FRACI_{FRAC_INPUTS_FIXED}_SEED_{SEED}_{time_stamp}'
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


w_e_e = 0.8e-3 / dt
w_e_i = w_e_e
w_i_e = -0.3e-4 / dt
w_e_e_added = 0.05 * w_e_e * 0.2

def make_network(e_i_scale):
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((n_e + n_i, n_e + n_i))

	w_initial[:n_e, :n_e] = 0.05 * w_e_e * (0.2 + 0.8 * np.random.rand(n_e, n_e))

	# w_initial[:n_e, :n_e] = np.diag(np.ones(n_e - 1), k=-1) * w_e_e * 0.7

	w_initial[n_e:, :n_e] = gaussian_if_under_val(0.8, (n_i, n_e), w_e_i * e_i_scale, 0.3 * w_e_i * e_i_scale)
	w_initial[n_e:, :n_e] = np.where(w_initial[n_e:, :n_e] < 0, 0, w_initial[n_e:, :n_e])
	w_initial[:n_e, n_e:] = gaussian_if_under_val(0.8, (n_e, n_i), w_i_e, 0.3 * np.abs(w_i_e))
	w_initial[:n_e, n_e:] = np.where(w_initial[:n_e, n_e:] > 0, 0, w_initial[:n_e, n_e:])

	np.fill_diagonal(w_initial, 0)
	return w_initial


def calc_loss(r : np.ndarray, train_times : np.ndarray, test_times : np.ndarray):

	if np.isnan(r).any():
		return 10000

	r_exc = r[:, :, :n_e]

	stacked_activities_train = []
	stacked_activities_test = []

	for i in range(r.shape[0]):
		if i < 6:
			stacked_activities_train.append(r_exc[i, train_times, :])
		else:
			stacked_activities_test.append(r_exc[i, test_times, :])

	X_train = np.concatenate(stacked_activities_train, axis=0)
	y_train = np.stack([train_times for j in range(6)]).flatten()

	X_test = np.concatenate(stacked_activities_test, axis=0)
	y_test = np.stack([test_times for j in range(r.shape[0] - 6)]).flatten()

	# print('X SHAPE', X.shape)
	# print('y SHAPE', y.shape)

	# print(X_train)
	# print(y_train)

	reg = LinearRegression().fit(X_train, y_train)

	# print(np.sum(r) / (r.shape[0] * r.shape[1] * r.shape[2]) * 100)

	loss = 1000 * (1 - reg.score(X_test, y_test)) + np.sum(r) / (r.shape[0] * r.shape[1] * r.shape[2]) * 100 # + np.sum(r[:, -1, :]) / (r.shape[0] * r.shape[2]) * 50e4

	print('loss:', loss)

	return loss


def plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=True):
	scale = 3
	n_res_to_show = BATCH_SIZE

	gs = gridspec.GridSpec(2 * n_res_to_show + 3, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 3) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 2, :])]

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

		for trial_idx in range(rs_for_loss.shape[0]):
			r = rs_for_loss[trial_idx, ...]

			for l_idx in range(r.shape[1]):
				if l_idx < n_e:
					if l_idx % 1 == 0:
						axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
						# axs[2 * i][0].plot(t, all_r_targets[loss_min_idx, :, l_idx], '--', c=layer_colors[l_idx % len(layer_colors)]) # graph target activity

						# axs[2 * i][0].plot(t, 4 * r_exp_filtered[:, l_idx], '-.', c=layer_colors[l_idx % len(layer_colors)]) # graph target activity
				else:
					axs[2 * i][1].plot(t, r[:, l_idx], c='black') # graph inh activity

		r_exc = r[:, :n_e]
		r_summed = np.sum(r_exc, axis=0)
		r_active_mask =  np.where(r_summed != 0, 1, 0).astype(bool)
		r_summed_safe_divide = np.where(r_active_mask, r_summed, 1)
		r_normed = r_exc / r_summed_safe_divide
		t_means = np.sum(t.reshape(t.shape[0], 1) * r_normed, axis=0)
		t_ordering = np.argsort(t_means)
		t_ordering = np.concatenate([t_ordering, np.arange(n_e, n_e + n_i)])

		sorted_w_initial = w_initial[t_ordering, :][:, t_ordering]
		sorted_w = w[t_ordering, :][:, t_ordering]

		vmin = np.min([w_initial.min(), w.min()])
		vmax = np.max([w_initial.max(), w.max()])

		vbound = np.maximum(vmax, np.abs(vmin))
		vbound = 5

		mappable = axs[2 * i + 1][0].matshow(sorted_w_initial, vmin=-vbound, vmax=vbound, cmap='coolwarm') # plot initial weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][0])

		mappable = axs[2 * i + 1][1].matshow(sorted_w, vmin=-vbound, vmax=vbound, cmap='coolwarm') # plot final weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][1])

		axs[2 * i][0].set_title(f'{true_losses[i]} + {syn_effect_penalties[i]}')
		for i_axs in range(2):
			axs[2 * i][i_axs].set_xlabel('Time (s)')
			axs[2 * i][i_axs].set_ylabel('Firing rate')

		axs[2 * n_res_to_show + 2].plot(np.arange(len(all_weight_deltas)), np.log(all_weight_deltas), label=f'{i}')

	partial_rules_len = int(len(plasticity_coefs))

	all_effects = np.array(all_effects)
	effects = np.mean(all_effects, axis=0)

	axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
	effects_argsort = []
	for l in range(1):
		effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
		effects_argsort_partial = np.flip(np.argsort(effects_partial))
		effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
		x = np.arange(len(effects_argsort_partial)) + l * partial_rules_len
		axs[2 * n_res_to_show + 1].bar(x, effects_partial[effects_argsort_partial], zorder=0)
		for i_e in x:
			axs[2 * n_res_to_show + 1].scatter(i_e * np.ones(all_effects.shape[0]), all_effects[:, effects_argsort_partial][:, i_e], c='black', zorder=1, s=3)
	axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
	axs[2 * n_res_to_show + 1].set_xlim(-1, len(effects))

	true_loss = np.sum(true_losses)
	syn_effect_penalty = np.sum(syn_effect_penalties)
	axs[2 * n_res_to_show].set_title(f'Loss: {true_loss + syn_effect_penalty}, {true_loss}, {syn_effect_penalty}')

	# plot the coefficients assigned to each plasticity rule (unsorted by size)
	for l in range(1):
		axs[2 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
	axs[2 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
	axs[2 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
	axs[2 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))

	axs[2 * n_res_to_show + 2].set_xlabel('Epochs')
	axs[2 * n_res_to_show + 2].set_ylabel('log(delta W)')
	axs[2 * n_res_to_show + 2].legend()

	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']
	prefix = eval_tracker['file_prefix']

	fig.tight_layout()
	if train:
		fig.savefig(f'{out_dir}/{prefix}_{zero_padding}{evals}.png')
	else:
		fig.savefig(f'{out_dir}/{prefix}_{zero_padding}{evals}_test.png')
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


def simulate_single_network(index, x, train, calibrate, track_params=True):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	if len(x) != N_RULES + N_TIMECONSTS + 1:
		raise ValueError('Improper length of input vector')

	print(x)

	plasticity_coefs = x[:N_RULES]
	rule_time_constants = x[N_RULES:-1]
	e_i_scale = x[-1]

	if FIXED_DATA:
		if train:
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network(e_i_scale) # make a new, distorted sequence

	decode_start = 3e-3/dt
	decode_end = 65e-3/dt
	train_times = (decode_start + np.random.rand(500) * (decode_end - decode_start - 1)).astype(int) # 500
	test_times = (decode_start + np.random.rand(200) * (decode_end - decode_start - 1)).astype(int)	# 200
	if calibrate:
		n_inner_loop_iters = 15
	else:
		n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	all_effects = np.zeros(plasticity_coefs.shape)
	normed_loss = 10000
	rs_for_loss = []

	w_hist = []
	all_weight_deltas = []
	w_hist.append(w)

	blew_up = False

	surviving_synapse_mask = np.ones((n_e, n_e)).astype(bool)

	fixed_inputs_spks = np.zeros((len(t), n_e + n_i))
	fixed_inputs_spks[:10, 0] = 1
	fixed_inputs_spks[10:int(decode_end), 1:n_e + n_i] = np.random.poisson(lam=INPUT_RATE_PER_CELL * FRAC_INPUTS_FIXED * dt, size=(int(decode_end) - 10, n_e - 1 + n_i))

	for i in range(n_inner_loop_iters):
		# Define input for activation of the network
		r_in = np.zeros((len(t), n_e + n_i))

		random_inputs_poisson = np.zeros((len(t), n_e + n_i))
		random_inputs_poisson[10:int(decode_end), :n_e + n_i] = np.random.poisson(lam=INPUT_RATE_PER_CELL * (1 - FRAC_INPUTS_FIXED) * dt, size=(int(decode_end) - 10, n_e + n_i))
		random_inputs_poisson[:, 0] = 0

		r_in = poisson_arrivals_to_inputs(fixed_inputs_spks + random_inputs_poisson, 3e-3)
		r_in[:, :n_e] = 0.09 * r_in[:, :n_e]
		r_in[:, -n_i:] = 0.02 * r_in[:, -n_i:]

		if i <= 1000:
			synapse_change_mask_for_i = np.random.rand(n_e, n_e) < CHANGE_PROB_PER_ITER

			drop_mask_for_i = np.logical_and(synapse_change_mask_for_i, surviving_synapse_mask)
			birth_mask_for_i = np.logical_and(synapse_change_mask_for_i, ~surviving_synapse_mask)

			surviving_synapse_mask[synapse_change_mask_for_i] = ~surviving_synapse_mask[synapse_change_mask_for_i]

			w[:n_e, :n_e] = np.where(drop_mask_for_i, 0, w[:n_e, :n_e])
			w[:n_e, :n_e] = np.where(birth_mask_for_i, w_e_e_added, w[:n_e, :n_e])

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects, r_exp_filtered = simulate(t, n_e, n_i, r_in, plasticity_coefs, rule_time_constants, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=1e-3, g=1, w_u=1, track_params=track_params)

		if np.isnan(r).any() or (np.abs(w_out) > 100).any() or (np.abs(w_out[:n_e, :n_e]) < 1.5e-6).all(): # if simulation turns up nans in firing rate matrix, end the simulation
			
			return {
				'blew_up': True,
				'syn_effects': all_effects,
			}

		if i in [n_inner_loop_iters - 1 - 1 * k for k in range(12)]:
			rs_for_loss.append(r)

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects[:N_RULES]

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	if i == n_inner_loop_iters - 1:
		rs_for_loss = np.stack(rs_for_loss)
		normed_loss = calc_loss(rs_for_loss, train_times, test_times)

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
	rule_time_constants = x[N_RULES:-1]
	e_i_scale = x[-1]

	if np.any(np.array([res['blew_up'] for res in results])):
		if eval_tracker is not None:
			eval_tracker['evals'] += 1
		syn_effects = np.stack([res['syn_effects'] for res in results])
		return 1e8 * BATCH_SIZE + 1e7 * np.sum(np.abs(plasticity_coefs)), 1e8 * np.ones((len(results),)), syn_effects

	true_losses = np.array([res['loss'] for res in results])
	syn_effects = np.stack([res['syn_effects'] for res in results])
	syn_effect_penalties = np.zeros(syn_effects.shape[0])
	one_third_len = int(syn_effects.shape[1])

	for i in range(1):
		syn_effect_penalties += L1_PENALTIES[i] * np.sum(np.abs(syn_effects[:, i * one_third_len:(i+1) * one_third_len]), axis=1)

	losses = true_losses + syn_effect_penalties
	loss = np.sum(losses)

	if eval_tracker is not None:
		if train:
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss'] or eval_tracker['file_prefix'] == 'CAL':
				if eval_tracker['evals'] > 0:
					eval_tracker['best_loss'] = loss
					eval_tracker['best_changed'] = True
					eval_tracker['params'] = copy(x)
				plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=True)
			eval_tracker['evals'] += 1
		else:
			plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=False)
			eval_tracker['best_changed'] = False

	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')
	return loss, true_losses, syn_effects


# def plasticity_coefs_eval_wrapper(plasticity_coefs, eval_tracker=None, track_params=False):
# 	if eval_tracker['evals'] > 0 and eval_tracker['evals'] % CALC_TEST_SET_LOSS_FREQ == 0 and eval_tracker['best_changed']:
# 		loss, true_losses, syn_effects = simulate_plasticity_rules(eval_tracker['plasticity_coefs'], eval_tracker=eval_tracker, track_params=track_params, train=False)
# 		eval_tracker['best_changed'] = False
# 		log_sim_results(test_data_path, eval_tracker, loss, true_losses, eval_tracker['plasticity_coefs'], syn_effects)

# 	loss, true_losses, syn_effects = simulate_plasticity_rules(plasticity_coefs, eval_tracker=eval_tracker, track_params=track_params, train=True)
# 	log_sim_results(train_data_path, eval_tracker, loss, true_losses, plasticity_coefs, syn_effects)
# 	return loss


def load_best_params(file_name):
	file_path = f'./sims_out/{file_name}/outcmaes/xrecentbest.dat'
	df_params = read_csv(file_path, read_header=False)
	x = np.arange(df_params.shape[0])
	min_loss_idx = np.argmin([df_params.iloc[i][4] for i in x])
	best_params = df_params.iloc[min_loss_idx][5:]
	return np.array(best_params)


def simulate_single_network_wrapper(tup):
	return simulate_single_network(*tup)


def eval_all(X, eval_tracker=None, train=True, calibrate=False):
	start = time.time()

	indices = np.arange(BATCH_SIZE)
	pool = mp.Pool(POOL_SIZE)

	task_vars = []
	for x in X:
		for idx in indices:
			task_vars.append((idx, x, train, calibrate))
	results = pool.map(simulate_single_network_wrapper, task_vars)

	pool.close()
	pool.join()

	losses = []
	all_syn_effects = []
	for i in range(len(X)):
		loss, true_losses, syn_effects = process_plasticity_rule_results(results[BATCH_SIZE * i: BATCH_SIZE * (i+1)], X[i], eval_tracker=eval_tracker, train=train)
		losses.append(loss)
		all_syn_effects.append(syn_effects)
		if train:
			log_sim_results(train_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
		else:
			log_sim_results(test_data_path, eval_tracker, loss, true_losses, X[i], syn_effects)
	
	dur = time.time() - start
	print('dur:', dur)

	all_syn_effects = np.array(all_syn_effects).sum(axis=(0, 1))

	return losses, all_syn_effects


def process_params_str(s):
	params = []
	for x in s.split(' '):
		x = x.replace('\n', '')
		if x != '':
			params.append(float(x))
	return np.array(params)


if __name__ == '__main__':
	mp.set_start_method('fork')

	# define the bounds for all parameters involved

	bounds = [
			np.array([-100] * N_RULES + [0.5e-3] * N_TIMECONSTS + [1e-5]),
			np.array([ 100] * N_RULES + [40e-3] * N_TIMECONSTS + [0.5]),
	]

	# bounds for benchmarking
	# bounds = [
	# 		np.array([-1] * (N_RULES +  N_TIMECONSTS + 1)),
	# 		np.array([+1] * (N_RULES +  N_TIMECONSTS + 1)),
	# ]

	scale_factors = (bounds[1] - bounds[0]) / 2
	biases = (bounds[1] + bounds[0]) / 2

	if args.load_initial is not None:
		x0 = (load_best_params(args.load_initial) - biases) * scale_factors
		x0[:N_RULES] *= STD_EXPL_RULES
		x0[N_RULES:N_RULES + N_TIMECONSTS] *= STD_EXPL_CONSTS
		x0[-1] *= STD_EXPL_INH
		x0 += biases
	else:
		# define x0 as vector of biases
		x0 = biases

	# # calibrate by tuning the scale of each rule

	# eval_tracker = {
	# 	'evals': 0,
	# 	'best_loss': np.nan,
	# 	'best_changed': False,
	# 	'file_prefix': 'CAL',
	# }

	# # define a rescaling based on the amount of synaptic impact / rule coefficient
	# rescalings = np.ones(N_RULES)
	# losses_cal = 1e8 * np.ones(N_RULES)
	# recal_indices = np.arange(N_RULES)

	# n = 0
	# while (np.array(losses_cal) > 1e7).any():
	# 	X_cal = []

	# 	for i_x in recal_indices:
	# 		X_cal.append(copy(biases))
	# 		X_cal[-1][i_x] += scale_factors[i_x] * STD_EXPL / np.power(10, n)

	# 	# measure the synaptic change due to each rule isolated

	# 	losses_cal, all_syn_effects_cal = eval_all(X_cal, eval_tracker=eval_tracker)

	# 	print(all_syn_effects_cal)

	# 	new_recal_indices = []

	# 	for i_x, loss_cal in zip(recal_indices, losses_cal):
	# 		if loss_cal < 1e7:
	# 			print('syn_effects', all_syn_effects_cal[i_x])
	# 			print('scale_factors[i_x]', scale_factors[i_x])
	# 			rescalings[i_x] = 1e-8 * all_syn_effects_cal[i_x] / (scale_factors[i_x] * STD_EXPL / np.power(10, n))
	# 		else:
	# 			new_recal_indices.append(i_x)

	# 	recal_indices = new_recal_indices

	# 	n += 1

	# print(rescalings)

	# rescalings = 1

	# define eval_tracker for actual optimization

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
		'file_prefix': '',
	}

	# define options for optimization

	base_options = {
		'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
		'bounds': [
			[-1] * (N_RULES + N_TIMECONSTS + 1),
			[1] * (N_RULES + N_TIMECONSTS + 1),
		],
		'tolfun': args.tol_func,
		'tolfunhist': args.tol_func_hist,
	}
	incpopsize = 2

	# eval_all = generate_surrogate_problem(N_RULES + N_TIMECONSTS + 1, 3)

	best = cma.optimization_tools.BestSolution()
	small_pop_fevals_count = 0
	large_pop_fevals_count = 0
	small_pop_runs_count = 0
	large_pop_runs_count = 0
	restarts = 0
	poptype = 'small'
	sigma_factor = 1

	es = None

	while True:
		options = copy(base_options)

		if restarts > 0:
			if small_pop_fevals_count < np.max([1, large_pop_fevals_count]):
				poptype = 'small'
				small_pop_runs_count += 1
				popsize_multiplier = incpopsize**(large_pop_runs_count)
				options['popsize'] = np.floor(popsize_multiplier**(np.random.uniform()**2) * base_options['popsize'])
				options['maxiter'] = np.min([base_options['maxiter'], np.floor(0.5 * large_pop_fevals_count / base_options['popsize'])])
				sigma_factor = 1e-2 * np.random.uniform()
			else:
				poptype = 'large'
				large_pop_runs_count += 1
				popsize_multiplier = incpopsize**(large_pop_runs_count)
				options['popsize'] = popsize_multiplier * base_options['popsize']
				options['maxiter'] = base_options['maxiter']

		print('biases:', biases)
		print('scale_factors:', scale_factors)
		print(args.std_expl)

		es = cma.CMAEvolutionStrategy(np.zeros(N_RULES + N_TIMECONSTS + 1), sigma_factor * 0.003, options)

		while not es.stop():
			X = es.ask()
			X_rescaled = []
			for i, x in enumerate(X):
				x_rescaled = copy(x) * scale_factors
				print('x:', x)
				x_rescaled[:N_RULES] *= STD_EXPL_RULES
				x_rescaled[N_RULES:N_RULES + N_TIMECONSTS] *= STD_EXPL_CONSTS
				x_rescaled[-1] *= STD_EXPL_INH
				x_rescaled += biases
				x_rescaled[bounds[0] > x_rescaled] = bounds[0][bounds[0] > x_rescaled]
				x_rescaled[bounds[1] < x_rescaled] = bounds[1][bounds[1] < x_rescaled]
				X_rescaled.append(x_rescaled)
			losses, all_syn_effects = eval_all(X_rescaled, eval_tracker=eval_tracker)
			es.tell(X, losses)
			if eval_tracker['best_changed']:
				eval_all([eval_tracker['params']], eval_tracker=eval_tracker, train=False)
			es.disp()

		if poptype == 'large':
			large_pop_fevals_count += es.countevals
			large_pop_runs_count += 1
		else:
			small_pop_fevals_count += es.countevals
			small_pop_runs_count += 1

		if restarts == 0:
			restarts = 1
			base_options['maxiter'] = es.opts['maxiter']
			base_options['popsize'] = es.opts['popsize']

		best.update(es.best, es.sent_solutions)
		print(best.get()[1])
		print(es.stop())

		if restarts >= N_RESTARTS:
			break

		print(f'RESTART {restarts}', f'POPTYPE {poptype}')
