from copy import deepcopy as copy
import numpy as np
import os
import time
import cma
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val, rev_argsort, set_smallest_n_zero
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
from sklearn.linear_model import LinearRegression
from csv_reader import read_csv
from csv_writer import write_csv

from rate_network import simulate, tanh, generate_gaussian_pulse

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='ttl', type=str)
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=3, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--frac_inputs_fixed', metavar='fi', type=float)
parser.add_argument('--syn_change_prob', metavar='cp', type=float, default=0.)
parser.add_argument('--seed', metavar='s', type=int)
parser.add_argument('--param_vec', metavar='ps', type=float, nargs=22)

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

TITLE = args.title
SEED = args.seed
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_RANGE = (600, 601) # Number of times to simulate network and plasticity rules per loss function evaluation
DECODER_TRAIN_ITERS = [569, 564, 559, 554, 549, 544]
DECODER_TEST_ITERS = [599, 594, 589, 584, 579, 574]
STD_EXPL = args.std_expl
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_JITTER_COEF = 60
CHANGE_PROB_PER_ITER = args.syn_change_prob #0.0007
FRAC_INPUTS_FIXED = args.frac_inputs_fixed
INPUT_RATE_PER_CELL = 80
N_RULES = 8
N_TIMECONSTS = 6
ACTIVE_RULES = np.flip(np.sort(np.nonzero(args.param_vec[-N_RULES:])[0])).astype(int)

print(ACTIVE_RULES)


T = 0.11 # Total duration of one network simulation
dt = 1e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
n_e = 40 # Number excitatory cells in sequence (also length of sequence)
n_i = 8 # Number inhibitory cells
train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
x_base = np.array(args.param_vec[:(N_RULES + N_TIMECONSTS)])

layer_colors = get_ordered_colors('gist_rainbow', 15)
np.random.shuffle(layer_colors)

rule_names = [ # Define labels for all rules to be run during simulations
	# r'',
	# r'$y$',
	# r'$x$',
	# r'$y^2$',
	# r'$x^2$',
	# r'$x \, y$',
	# r'$\tilde{y}$',
	r'$x \, \tilde{y}$',
	# r'$\tilde{x}$',
	r'$\tilde{x} \, y$',
	# r'$x_{int} \, y^2$',
	r'$\tilde{y} \, y$',
	# r'$\tilde{x} \, x$',
	# r'$\tilde{y}^2$',
	# r'$\tilde{x}^2$',

	# r'$w$',
	# r'$w y$',
	# r'$w x$',
	# r'$y^2$',
	# r'$x^2$',
	# r'$w x \, y$',
	# r'$w \tilde{y}$',
	r'$w x \, \tilde{y}$',
	# r'$w \tilde{x}$',
	r'$w \tilde{x} \, y$',
	# r'$x_{int} \, y^2$',
	r'$w \tilde{y} \, y$',
	# r'$w \tilde{x} \, x$',
	# r'$w \tilde{y}^2$',
	# r'$w \tilde{x}^2$',

	# r'$w^2$',
	# r'$w^2 \, y$',
	# r'$w^2 \, x$',
	# r'$w^2 \, y^2$',
	# r'$w^2 \, x^2$',
	# r'$w^2 \, x \, y$',
	# r'$w^2 \, x \, y^2$',
	# r'$w^2 \, x^2 \, y$',
	# r'$w^2 \, x^2 \, y^2$',
	# r'$w^2 y_{int}$',
	# r'$w^2 x \, y_{int}$',
	# r'$w^2 x_{int}$',
	# r'$w^2 x_{int} \, y$',
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
out_dir = f'sims_out/kernel_test_indep_scale_long_{TITLE}_BATCH_{BATCH_SIZE}_FIXED_{FIXED_DATA}_L1_PENALTY_{joined_l1}_CHANGEP_{CHANGE_PROB_PER_ITER}_FRACI_{FRAC_INPUTS_FIXED}_SEED_{SEED}_{time_stamp}'
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
w_e_i = 0.5e-4 / dt
w_i_e = -0.3e-4 / dt
w_e_e_added = 0.05 * w_e_e * 0.2

def create_shift_matrix(size, k=1, weighting=[]):
	w = np.zeros((size, size))
	if k >= 1:
		for k_p in np.arange(1, k+1):
			a = 1
			if len(weighting) == k:
				a = weighting[k_p - 1]
			w += a * np.diag(np.ones((size - k_p,)), k=k_p)
			# w[(size - k_p):, k - k_p] = 1

	elif k <= -1:
		for k_p in np.arange(1, -k+1):
			a = 1
			if len(weighting) == -k:
				a = weighting[k_p - 1]
			w += a * np.diag(np.ones((size - k_p,)), k=-k_p)
			# w[-k - k_p, (size - k_p):] = 1
	return w

def make_network():
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((n_e + n_i, n_e + n_i))

	w_initial[:n_e, :n_e] = w_e_e * (0.05 * (0.2 + 0.8 * np.random.rand(n_e, n_e)))

	# w_initial[:n_e, :n_e][np.random.rand(n_e, n_e) >= 0.8] = 0

	# w_initial[:n_e, :n_e] = np.diag(np.ones(n_e - 1), k=-1) * w_e_e * 0.7

	w_initial[n_e:, :n_e] = gaussian_if_under_val(1, (n_i, n_e), w_e_i, 0 * w_e_i)
	w_initial[n_e:, :n_e] = np.where(w_initial[n_e:, :n_e] < 0, 0, w_initial[n_e:, :n_e])
	w_initial[:n_e, n_e:] = gaussian_if_under_val(1, (n_e, n_i), w_i_e, 0 * np.abs(w_i_e))
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

	print(X_train)
	print(y_train)

	reg = LinearRegression().fit(X_train, y_train)

	print(np.sum(r) / (r.shape[0] * r.shape[1] * r.shape[2]) * 100)

	loss = 1000 * (1 - reg.score(X_test, y_test)) + np.sum(r) / (r.shape[0] * r.shape[1] * r.shape[2]) * 100

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

			write_csv(os.path.join(out_dir, f'all_r_{eval_tracker["evals"]}.csv'), rs_for_loss[trial_idx, :, :n_e], delimiter=',')

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

		write_csv(os.path.join(out_dir, f'weight_mat_{eval_tracker["evals"]}.csv'), w, delimiter=',')


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
	plasticity_coefs = x[:N_RULES - 2]
	weight_bounds = x[N_RULES - 2 : N_RULES]
	rule_time_constants = x[N_RULES:(N_RULES + N_TIMECONSTS)]

	if FIXED_DATA:
		if train:
			print(train_seeds[index])
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network() # make a new, distorted sequence

	print(np.mean(w_initial))

	decode_start = 3e-3/dt
	decode_end = 105e-3/dt
	train_times = (decode_start + np.random.rand(500) * (decode_end - decode_start - 1)).astype(int) # 500
	test_times = (decode_start + np.random.rand(200) * (decode_end - decode_start - 1)).astype(int)	# 200
	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	all_effects = np.zeros(plasticity_coefs.shape)
	normed_loss = 10000
	rs_for_loss = []

	w_hist = []
	all_weight_deltas = []
	w_hist.append(w)

	# surviving_synapse_mask = np.ones((n_e, n_e)).astype(bool)

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

		# if i >= 400 and i < 800:
		# 	synapse_change_mask_for_i = np.random.rand(n_e, n_e) < CHANGE_PROB_PER_ITER

		# 	drop_mask_for_i = np.logical_and(synapse_change_mask_for_i, surviving_synapse_mask)
		# 	birth_mask_for_i = np.logical_and(synapse_change_mask_for_i, ~surviving_synapse_mask)

		# 	surviving_synapse_mask[synapse_change_mask_for_i] = ~surviving_synapse_mask[synapse_change_mask_for_i]

		# 	w[:n_e, :n_e] = np.where(drop_mask_for_i, 0, w[:n_e, :n_e])
		# 	w[:n_e, :n_e] = np.where(birth_mask_for_i, w_e_e_added, w[:n_e, :n_e])

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects, r_exp_filtered = simulate(t, n_e, n_i, r_in, plasticity_coefs, rule_time_constants, weight_bounds, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

		if np.isnan(r).any() or (np.abs(w_out) > 100).any() or (np.abs(w_out[:n_e, :n_e]) < 1.5e-6).all(): # if simulation turns up nans in firing rate matrix, end the simulation
			return {
				'blew_up': True,
			}
			

		if (i in DECODER_TRAIN_ITERS) or (i in DECODER_TEST_ITERS):
			rs_for_loss.append(r)

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects[:N_RULES - 2]

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
	plasticity_coefs = x[:N_RULES - 2]
	weight_bounds = x[N_RULES - 2:N_RULES]
	rule_time_constants = x[N_RULES:(N_RULES + N_TIMECONSTS)]

	if np.any(np.array([res['blew_up'] for res in results])):
		if eval_tracker is not None:
			eval_tracker['evals'] += 1
		return 1e8 * BATCH_SIZE + 1e7 * np.sum(np.abs(plasticity_coefs)), 1e8 * np.ones((len(results),)), np.zeros((len(results), len(plasticity_coefs)))

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
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
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


def eval_all(X, eval_tracker=None, train=True):
	start = time.time()

	indices = np.arange(BATCH_SIZE)
	pool = mp.Pool(POOL_SIZE)

	task_vars = []
	for i_x, x in enumerate(X):
		for idx in indices:
			task_vars.append((idx, x, train))
	results = pool.map(simulate_single_network_wrapper, task_vars)

	pool.close()
	pool.join()

	losses = []
	for i in range(len(X)):
		loss, true_losses, syn_effects = process_plasticity_rule_results(results[BATCH_SIZE * i: BATCH_SIZE * (i+1)], X[i], eval_tracker=eval_tracker, train=train)
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

if __name__ == '__main__':
	mp.set_start_method('fork')
	
	rule_contingency_map = [
		[0, 8],
		[1, 9],
		[2, 10],
		[3, 11],
		[4, 12],
		[5, 13],
		[6],
		[7],
	]

	activated_terms = np.sort(np.concatenate([rule_contingency_map[r_idx] for r_idx in ACTIVE_RULES]))
	x0 = copy(x_base)[activated_terms]

	x_scale = np.abs(x0)

	a0 = x0 / x_scale

	print(a0)

	coefs_lower_bounds = [(-10 * int(x0[coef_idx] < 0)) for coef_idx in range(len(ACTIVE_RULES))]
	coefs_upper_bounds = [(10 * int(x0[coef_idx] >= 0)) for coef_idx in range(len(ACTIVE_RULES))]

	bounds = [
		coefs_lower_bounds + [0.5e-3] * (len(x0) - len(ACTIVE_RULES)),
		coefs_upper_bounds + [40e-3] * (len(x0) - len(ACTIVE_RULES)),
	]

	for i_b in range(len(bounds)):
		bounds[i_b] = (np.array(bounds[i_b]) / x_scale).tolist()

	eval_tracker = {
		'evals': 0,
		'best_loss': np.nan,
		'best_changed': False,
	}

	options = {
		'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
		# 'popsize': 15,
		'bounds': bounds,
	}

	# coefs = args.param_vec[:-2]
	# bounds = args.param_vec[-2:]

	# x = np.concatenate([coefs, time_consts, bounds])

	# w-STDP + summed weight bound
	# x = np.concatenate([0.3 * np.array([-0.0001, 0.0001, 0, -0.005, 0.005, 0]), time_consts, [7.5, 7.5]])

	# STDP + summed weight bound
	# x = np.concatenate([0.3 * np.array([-0.03, 0.03, 0, 0, 0, 0]), time_consts, [7.5, 7.5]])

	# successful rules for STDP + firing rate bound
	# x = np.concatenate([0.3 * np.array([-0.03, 0.03, 0, 0, 0, -0.005]), time_consts, [1000, 1000]])

	# successful rules for w-STDP + firing rate bound
	# x = np.concatenate([0.1 * np.array([0, 0, 0, -0.01, 0.01, -0.005]), time_consts, [1000, 1000]])


	# time_consts = np.array([6e-3, 6e-3, 6e-3, 6e-3, 0.5e-3, 0.5e-3])
	# x_ila = np.concatenate([0.1 * np.array([0, 0, 0, -0.01, 0.01, -0.005]), time_consts, [1000, 1000]])

	# time_consts = np.array([6e-3, 6e-3, 6e-3, 6e-3, 0.5e-3, 0.5e-3])
	# x_ila = np.concatenate([0.3 * np.array([-0.03, 0.03, 0, 0, 0, -0.005]), time_consts, [1000, 1000]])


	# x = np.concatenate([0.1 * np.array([0, 0, 0, 0, 0, 0]), time_consts, [1000, 1000]])

	es = cma.CMAEvolutionStrategy(a0, STD_EXPL, options)
	while not es.stop():
		A = es.ask()
		X_expanded = [copy(x_base) for k_sample in range(len(A))]

		len(activated_terms)

		for a_idx, a in enumerate(A):
			X_expanded[a_idx][activated_terms] = (np.array(a) * x_scale).tolist()

		print(X_expanded[0])

		es.tell(A, eval_all(X_expanded, eval_tracker=eval_tracker))
		if eval_tracker['best_changed']:
			eval_all([eval_tracker['params']], eval_tracker=eval_tracker, train=False)
		es.disp()
