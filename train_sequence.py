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
from csv_reader import read_csv
from csv_writer import write_csv

from rate_network import simulate, tanh, generate_gaussian_pulse

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, nargs=3, help='Prefactor for L1 penalties on loss function')
parser.add_argument('--asp', metavar='asp', type=int, help="Flag for penalizing shape of neurons' activity ")
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')
parser.add_argument('--fixed_data', metavar='fd', type=int, help='')
parser.add_argument('--load_initial', metavar='li', type=str, help='File from which to load the best params as an initial guess')

args = parser.parse_args()
print(args)

POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_RANGE = (190, 200) # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL = args.std_expl
DW_LAG = 5
FIXED_DATA = bool(args.fixed_data)
L1_PENALTIES = args.l1_pen
CALC_TEST_SET_LOSS_FREQ = 11
ACTIVITY_LOSS_COEF = 0.1 if bool(args.asp) else 0
ACTIVITY_JITTER_COEF = 3

T = 0.1 # Total duration of one network simulation
dt = 1e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
n_e = 15 # Number excitatory cells in sequence (also length of sequence)
n_i = 20 # Number inhibitory cells
train_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)
test_seeds = np.random.randint(0, 1e7, size=BATCH_SIZE)

layer_colors = get_ordered_colors('winter', 15)

rule_names = [ # Define labels for all rules to be run during simulations
	r'',
	r'$y$',
	r'$x$',
	r'$y^2$',
	# r'$x^2$',
	r'$x \, y$',
	r'$x \, y^2$',
	r'$x^2 \, y$',
	# r'$x^2 \, y^2$',
	# r'$y_{int}$',
	# r'$x \, y_{int}$',
	# r'$x_{int}$',
	r'$x_{int} \, y$',
	r'$x_{int} \, y^2$',

	r'$w$',
	r'$w \, y$',
	r'$w \, x$',
	r'$w \, y^2$',
	# r'$w \, x^2$',
	r'$w \, x \, y$',
	r'$w \, x \, y^2$',
	r'$w \, x^2 \, y$',
	# r'$w \, x^2 \, y^2$',
	# r'$w y_{int}$',
	# r'$w x \, y_{int}$',
	# r'$w x_{int}$',
	r'$w x_{int} \, y$',
	r'$w x_{int} \, y^2$',

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
out_dir = f'sims_out/seq_ee_jitter_pen_batch_{BATCH_SIZE}_STD_EXPL_{STD_EXPL}_FIXED_{FIXED_DATA}_L1_PENALTY_{joined_l1}_ACT_PEN_{args.asp}_{time_stamp}'
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

def make_network():
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((n_e + n_i, n_e + n_i))
	w_initial[:n_e, :n_e] = w_e_e * np.diag(0.8 * np.log10(np.arange(n_e - 1) + 10), k=-1)
	w_initial[:n_e, :n_e] = w_initial[:n_e, :n_e] * (0.3 + 1.4  * np.random.rand(n_e, n_e))

	w_initial[:n_e, :n_e] = np.where(
		np.diag(np.ones(n_e - 1), k=-1) > 0,
		w_initial[:n_e, :n_e],
		exp_if_under_val(0.5, (n_e, n_e), 0.03 * w_e_e)
	)

	w_initial[n_e:, :n_e] = gaussian_if_under_val(0.8, (n_i, n_e), w_e_i, 0.3 * w_e_i)
	w_initial[:n_e, n_e:] = gaussian_if_under_val(0.8, (n_e, n_i), w_i_e, 0.3 * np.abs(w_i_e))

	np.fill_diagonal(w_initial, 0)
	return w_initial


def calc_loss(r : np.ndarray):

	if np.isnan(r).any():
		return 1e8

	r_exc = r[:, :n_e]

	r_summed = np.sum(r_exc, axis=0)
	r_active_mask =  np.where(r_summed != 0, 1, 0).astype(bool)
	r_summed_safe_divide = np.where(r_active_mask, r_summed, 1)

	r_normed = r_exc / r_summed_safe_divide
	t_means = np.sum(t.reshape(t.shape[0], 1) * r_normed, axis=0)
	t_vars = np.sum(np.power(t.reshape(t.shape[0], 1), 2) * r_normed, axis=0) - np.power(t_means, 2)

	# print('r_summed', r_summed)
	# print('t_means:', t_means)
	# print('t_vars:', t_vars)

	# add measure of spike consistency

	t_means_nan = copy(t_means)
	t_means_nan[~r_active_mask] = np.nan
	t_means_diffs = t_means_nan[1:] - t_means_nan[:-1]

	loss = 0
	activity_loss = 0

	for i in np.arange(r_exc.shape[1]):
		if r_active_mask[i]:
			if i != 0:
				loss_activity_var = ACTIVITY_LOSS_COEF * np.power((t_vars[i] - 4e-6) / 4e-6, 2)
				loss += loss_activity_var
				activity_loss += loss_activity_var

				loss_activity_zero_mom = ACTIVITY_LOSS_COEF * np.power((r_summed[i] - 15) / 15, 2)
				loss += loss_activity_zero_mom
				activity_loss += loss_activity_zero_mom

			if i < (r_exc.shape[1] - 1) and r_active_mask[i+1]:
				loss += 1 / (1 + np.exp((t_means[i+1] - t_means[i] - 5e-4) / 1e-4))
		else:
			loss += 100

	return loss, t_means_diffs, activity_loss


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
		r, w, w_initial, normed_loss, effects, all_weight_deltas, r_exp_filtered = results[i]

		all_effects.append(effects)

		for l_idx in range(r.shape[1]):
			if l_idx < n_e:
				if l_idx % 1 == 0:
					axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
					# axs[2 * i][0].plot(t, all_r_targets[loss_min_idx, :, l_idx], '--', c=layer_colors[l_idx % len(layer_colors)]) # graph target activity

					# axs[2 * i][0].plot(t, 4 * r_exp_filtered[:, l_idx], '-.', c=layer_colors[l_idx % len(layer_colors)]) # graph target activity
			else:
				axs[2 * i][1].plot(t, r[:, l_idx], c='black') # graph inh activity

		vmin = np.min([w_initial.min(), w.min()])
		vmax = np.max([w_initial.max(), w.max()])

		mappable = axs[2 * i + 1][0].matshow(w_initial, vmin=vmin, vmax=vmax) # plot initial weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][0])

		mappable = axs[2 * i + 1][1].matshow(w, vmin=vmin, vmax=vmax) # plot final weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][1])

		axs[2 * i][0].set_title(f'{true_losses[i]} + {syn_effect_penalties[i]}')
		for i_axs in range(2):
			axs[2 * i][i_axs].set_xlabel('Time (s)')
			axs[2 * i][i_axs].set_ylabel('Firing rate')

		axs[2 * n_res_to_show + 2].plot(np.arange(len(all_weight_deltas)), np.log(all_weight_deltas), label=f'{i}')

	### plot the coefficients assigned to each plasticity rule
	# plasticity_coefs_abs = np.abs(plasticity_coefs)
	# plasticity_coefs_argsort = np.flip(np.argsort(plasticity_coefs_abs))
	# axs[2 * n_res_to_show + 1].bar(np.arange(len(plasticity_coefs)), plasticity_coefs[plasticity_coefs_argsort])
	# axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(plasticity_coefs)))
	# axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[plasticity_coefs_argsort], rotation=60, ha='right')
	# axs[2 * n_res_to_show + 1].set_xlim(-1, len(plasticity_coefs))
	partial_rules_len = int(len(plasticity_coefs))

	all_effects = np.array(all_effects)
	print('effects shape', all_effects.shape)
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


def simulate_single_network(index, plasticity_coefs, track_params=False, train=True):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	if FIXED_DATA:
		if train:
			np.random.seed(train_seeds[index])
		else:
			np.random.seed(test_seeds[index])
	else:
		np.random.seed()

	w_initial = make_network() # make a new, distorted sequence
	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	cumulative_loss = 0

	all_effects = np.zeros(plasticity_coefs.shape)
	all_mean_active_time_diffs = []
	total_activity_loss = 0

	w_hist = []
	all_weight_deltas = []
	w_hist.append(w)

	for i in range(n_inner_loop_iters):
		# Define input for activation of the network
		r_in = np.zeros((len(t), n_e + n_i))
		input_amp = np.random.rand() * 0.002 + 0.01
		r_in[:, 0] = generate_gaussian_pulse(t, 5e-3, 5e-3, w=input_amp) # Drive first excitatory cell with Gaussian input

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects, r_exp_filtered = simulate(t, n_e, n_i, r_in + 4e-6 / dt * np.random.rand(len(t), n_e + n_i), plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

		if i in [n_inner_loop_iters - 1 - 10 * k for k in range(5)]:
			# loss_start = time.time()
			loss, mean_active_time_diffs, activity_loss = calc_loss(r)
			all_mean_active_time_diffs.append(mean_active_time_diffs)
			total_activity_loss += activity_loss

			# print(time.time() - loss_start)
			cumulative_loss += loss

		if np.isnan(r).any(): # if simulation turns up nans in firing rate matrix, end the simulation
			cumulative_loss += 1e8
			break

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	normed_loss = cumulative_loss / 5

	if i == n_inner_loop_iters - 1:
		all_mean_active_time_diffs = np.stack(all_mean_active_time_diffs)
		mean_active_time_stds = np.nanstd(all_mean_active_time_diffs, axis=0)
		mean_active_time_means = np.nanmean(all_mean_active_time_diffs, axis=0)

		non_nan_non_zero = np.bitwise_and(~np.isnan(mean_active_time_means), mean_active_time_means != 0)
		mean_active_time_normed_stds = np.where(non_nan_non_zero, mean_active_time_stds / mean_active_time_means, 0)

		print('jitter_penalty:', ACTIVITY_JITTER_COEF * np.sum(mean_active_time_normed_stds))
		print('activity_loss:', total_activity_loss / 5)
		normed_loss += ACTIVITY_JITTER_COEF * np.sum(mean_active_time_normed_stds)


	return r, w, w_initial, normed_loss, all_effects, all_weight_deltas, r_exp_filtered


def log_sim_results(write_path, eval_tracker, loss, true_losses, plasticity_coefs, syn_effects):
	# eval_num, loss, true_losses, plastic_coefs, syn_effects
	syn_effect_means = np.mean(syn_effects, axis=0)
	syn_effect_stds = np.std(syn_effects, axis=0)
	to_save = np.concatenate([[eval_tracker['evals'], loss], true_losses, plasticity_coefs, syn_effect_means, syn_effect_stds]).flatten()
	print(to_save)
	write_csv(write_path, list(to_save))


# Function to minimize (including simulation)
def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None, track_params=False, train=True):
	start = time.time()

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs, track_params=track_params, train=train)
	results = pool.map(f, np.arange(BATCH_SIZE))
	pool.close()

	true_losses = np.array([res[3] for res in results])
	syn_effects = np.stack([res[4] for res in results])
	syn_effect_penalties = np.zeros(syn_effects.shape[0])
	one_third_len = int(syn_effects.shape[1] / 3)

	for i in range(3):
		syn_effect_penalties += L1_PENALTIES[i] * np.sum(np.abs(syn_effects[:, i * one_third_len:(i+1) * one_third_len]), axis=1)

	losses = true_losses + syn_effect_penalties
	loss = np.sum(losses)

	if eval_tracker is not None:
		if train:
			if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
				if eval_tracker['evals'] > 0:
					eval_tracker['best_loss'] = loss
					eval_tracker['best_changed'] = True
					eval_tracker['plasticity_coefs'] = copy(plasticity_coefs)
				plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=True)
			eval_tracker['evals'] += 1
		else:
			plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, train=False)

	dur = time.time() - start
	print('duration:', dur)
	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')
	return loss, true_losses, syn_effects


def plasticity_coefs_eval_wrapper(plasticity_coefs, eval_tracker=None, track_params=False):
	if eval_tracker['evals'] > 0 and eval_tracker['evals'] % CALC_TEST_SET_LOSS_FREQ == 0 and eval_tracker['best_changed']:
		loss, true_losses, syn_effects = simulate_plasticity_rules(plasticity_coefs, eval_tracker=eval_tracker, track_params=track_params, train=False)
		eval_tracker['best_changed'] = False
		log_sim_results(test_data_path, eval_tracker, loss, true_losses, eval_tracker['plasticity_coefs'], syn_effects)

	loss, true_losses, syn_effects = simulate_plasticity_rules(plasticity_coefs, eval_tracker=eval_tracker, track_params=track_params, train=True)
	log_sim_results(train_data_path, eval_tracker, loss, true_losses, plasticity_coefs, syn_effects)
	return loss


def load_best_params(file_name):
	file_path = f'./sims_out/{file_name}/outcmaes/xrecentbest.dat'
	df_params = read_csv(file_path, read_header=False)
	x = np.arange(df_params.shape[0])
	min_loss_idx = np.argmin([df_params.iloc[i][4] for i in x])
	best_params = df_params.iloc[min_loss_idx][5:]
	return np.array(best_params)


# x1_raw = """-0.00010793665215095119 -0.0077226235932765 -0.0007816865595492941 -0.03318025977609449 0.008092421601561416 -6.407613106235442e-05 8.610610693006271e-05 4.427703899099249e-05 -0.05225696475640676 0.10582698186377604 -0.05720473550010104 0.000190178600002215 -0.0016783840840024033 0.0028419457171027927"""
# print(x1_raw)

# def process_params_str(s):
# 	params = []
# 	for x in s.split(' '):
# 		x = x.replace('\n', '')
# 		if x is not '':
# 			params.append(float(x))
# 	return np.array(params)

# x1 = process_params_str(x1_raw)
# x1 = np.array([0, 0, 0, 0, 0, 0, 0, 0.00450943, 0, 0, -0.03998377, 0, -0.05242701, 0.0052274])

if args.load_initial is not None:
	x0 = load_best_params(args.load_initial)
else:
	x0 = np.zeros(18)

# x0[-1] = 0.005
# x0[-4] = -0.06

print(x0)

eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
	'best_changed': False,
}

plasticity_coefs_eval_wrapper(x0, eval_tracker=eval_tracker, track_params=True)

options = {
	'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
}

x, es = cma.fmin2(
	partial(plasticity_coefs_eval_wrapper, eval_tracker=eval_tracker, track_params=True),
	x0,
	STD_EXPL,
	restarts=10,
	bipop=True,
	options=options)

print(x)
print(es.result_pretty())
