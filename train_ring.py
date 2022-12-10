from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
from sklearn.decomposition import PCA

from rate_network import simulate, tanh, generate_gaussian_pulse

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, help='Prefactor for L1 penalty on loss function')
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')

args = parser.parse_args()
print(args)

POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
INPUT_NUM_PER_NTWK = 1
N_INNER_LOOP_RANGE = (150, 300) # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL = args.std_expl
L1_PENALTY = args.l1_pen
MCP_GAMMA = 2e-4

T = 0.1
dt = 1e-4
t = np.linspace(0, T, int(T / dt))
n_e = 15
n_i = 20

if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
out_dir = f'sims_out/ring_STD_EXPL_{STD_EXPL}_L1_PENALTY_{L1_PENALTY}_{time_stamp}'
os.mkdir(out_dir)
os.mkdir(os.path.join(out_dir, 'outcmaes'))

layer_colors = get_ordered_colors('winter', 15)

rule_names = [ # Define labels for all rules to be run during simulations
	r'',
	r'$y$',
	# r'$x$',
	r'$y^2$',
	# r'$x^2$',
	r'$x \, y$',
	r'$x \, y^2$',
	# r'$x^2 \, y$',
	# r'$x^2 \, y^2$',
	r'$y_{int}$',
	# r'$x \, y_{int}$',
	r'$x_{int}$',
	r'$x_{int} \, y$',

	r'$w$',
	r'$w \, y$',
	# r'$w \, x$',
	r'$w \, y^2$',
	# r'$w \, x^2$',
	r'$w \, x \, y$',
	r'$w \, x \, y^2$',
	# r'$w \, x^2 \, y$',
	# r'$w \, x^2 \, y^2$',
	r'$w y_{int}$',
	# r'$w x \, y_{int}$',
	r'$w x_{int}$',
	r'$w x_{int} \, y$',

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

transfer_e = partial(tanh, v_th=0.1)
transfer_i = partial(tanh, v_th=0.1)

w_e_e = 0.2e-3 / dt
w_e_i = 0.5e-4 / dt
w_i_e = -0.4e-4 / dt

r_in_e_0 = np.zeros((len(t), n_e))
r_target_0 = np.stack([[0.15149132, 0.11786643, 0.02162234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02162234, 0.11786643] for k in range(100)])
r_in_e_0[:100, :] = copy(r_target_0)

all_r_in = []
all_r_target = []

for i in range(n_e):
	all_r_in.append(np.concatenate([np.roll(r_in_e_0, i, axis=1), np.zeros((len(t), n_i))], axis=1))
	all_r_target.append(np.roll(r_target_0, i, axis=1))

def make_network():
	w_initial = np.zeros((n_e + n_i, n_e + n_i))
	ring_connectivity = w_e_e * 0.5 * (1 + np.cos(2 * np.pi / n_e * np.arange(n_e)))
	for r_idx in np.arange(n_e):
		w_initial[r_idx:n_e, r_idx] = ring_connectivity[:(n_e - r_idx)]
		w_initial[:r_idx, r_idx] = ring_connectivity[(n_e - r_idx):]

	w_initial[:n_e, :n_e] = w_initial[:n_e, :n_e] * (0.6 + 0.4 * np.random.rand(n_e, n_e))

	w_initial[n_e:, :n_e] = gaussian_if_under_val(0.8, (n_i, n_e), w_e_i, 0.3 * w_e_i)
	w_initial[:n_e, n_e:] = gaussian_if_under_val(0.8, (n_e, n_i), w_i_e, 0.3 * np.abs(w_i_e))

	return w_initial

def l2_loss(r, r_target):
	if np.isnan(r).any():
		return 1e8
	return np.sum(np.square(r[(-1 * r_target.shape[0]):, :n_e] - r_target))

def plot_results(results, eval_tracker, out_dir, title, plasticity_coefs):
	scale = 3
	n_res_to_show = INPUT_NUM_PER_NTWK * BATCH_SIZE

	gs = gridspec.GridSpec(2 * n_res_to_show + 2, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 2) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]

	for i in range(n_res_to_show):
		r, w, w_initial, loss = results[i]

		axs[2 * i][0].imshow(r[:, :n_e].T, aspect='auto', interpolation='none')

		for l_idx in range(r.shape[1]):
			if l_idx >= n_e:
				axs[2 * i][1].plot(t, r[:, l_idx], c='black')

		axs[2 * i + 1][0].matshow(w_initial)
		axs[2 * i + 1][1].matshow(w)
		axs[2 * i][0].set_title(title)

	partial_rules_len = int(len(plasticity_coefs) / 3)

	# axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
	# effects_argsort = []
	# for l in range(3):
	# 	effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
	# 	effects_argsort_partial = np.flip(np.argsort(effects_partial))
	# 	effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
	# 	axs[2 * n_res_to_show + 1].bar(np.arange(len(effects_argsort_partial)) + l * 16, effects_partial[effects_argsort_partial] / np.max(np.abs(effects_partial)))
	# axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
	# axs[2 * n_res_to_show + 1].set_xlim(-1, len(effects))

	# plot the coefficients assigned to each plasticity rule (unsorted by size)
	for l in range(3):
		axs[2 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
	axs[2 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
	axs[2 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
	axs[2 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))


	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	fig.tight_layout()
	fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	plt.close('all')

def simulate_single_network(args, plasticity_coefs, gamma=0.98):
	index = args[0]
	np.random.seed()

	w_initial = make_network()
	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	cumulative_loss = 0

	for i in range(n_inner_loop_iters):
		r_in_idx = np.random.randint(0, len(all_r_in))
		r_in = all_r_in[r_in_idx]

		r, s, v, w_out, effects = simulate(t, n_e, n_i, r_in + 2e-6 / dt * np.random.rand(len(t), n_e + n_i), transfer_e, transfer_i, plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1.5)
		
		loss = l2_loss(r, all_r_target[r_in_idx])
		cumulative_loss = cumulative_loss * gamma + loss

		if np.isnan(r).any():
			break
		w = w_out

	return r, w, w_initial, cumulative_loss / (1 / np.log(1/gamma))

def mcp_penalty(plasticity_coefs):
	coefs_abs = np.abs(plasticity_coefs)
	mcp_thresh = L1_PENALTY * MCP_GAMMA
	return L1_PENALTY * np.sum(np.where(coefs_abs >= mcp_thresh, mcp_thresh / 2, coefs_abs - np.square(coefs_abs) / (2 * mcp_thresh)))

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None):
	start = time.time()

	input_indices_to_test = np.stack([np.random.choice(np.arange(n_e), size=INPUT_NUM_PER_NTWK, replace=False) for i in range(BATCH_SIZE)])

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs)
	args = []
	for i in range(BATCH_SIZE * INPUT_NUM_PER_NTWK):
		args.append((i,))
	results = pool.map(f, args)
	pool.close()

	loss = np.sum([res[3] for k, res in enumerate(results)]) + BATCH_SIZE * mcp_penalty(plasticity_coefs)

	if eval_tracker is not None:
		if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
			if eval_tracker['evals'] > 0:
				eval_tracker['best_loss'] = loss
			plot_results(results, eval_tracker, out_dir, f'Loss: {loss}\n', plasticity_coefs)
		eval_tracker['evals'] += 1

	dur = time.time() - start
	print('duration:', dur)
	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')

	return loss

eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
}

x0 = np.zeros(48)

simulate_plasticity_rules(x0, eval_tracker=eval_tracker)

options = {
	'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
}

x, es = cma.fmin2(partial(simulate_plasticity_rules, eval_tracker=eval_tracker), x0, STD_EXPL, options=options)
print(x)
print(es.result_pretty())
