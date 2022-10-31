from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp

import cma
from sklearn.decomposition import PCA

from rate_network import simulate_seq, tanh, generate_gaussian_pulse

N_NETWORKS = 10
POOL_SIZE = 8
N_INNER_LOOP_ITERS = 200

print(cma.CMAOptions('verb'))

T = 0.1
dt = 1e-4
t = np.linspace(0, T, int(T / dt))
n_e = 15
n_i = 10

if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

out_dir = f'sims_out/{datetime.now()}'
os.mkdir(out_dir)
os.mkdir(os.path.join(out_dir, 'outcmaes'))

layer_colors = get_ordered_colors('winter', 15)

r_in = generate_gaussian_pulse(t, 0.005, 0.005, w=1)

transfer_e = partial(tanh, v_th=0.1)
transfer_i = partial(tanh, v_th=0.5)

plasticity_coefs = np.zeros(81)

w_e_e = 4e-4 / dt
w_e_i = 1e-4 / dt
w_i_e = -2.5e-5 / dt

all_w_initial = []

for i in range(N_NETWORKS):
	w_initial = np.zeros((n_e + n_i, n_e + n_i))
	w_initial[:n_e, :n_e] = np.where(
		np.diag(np.ones(n_e - 1), k=-1) > 0,
		gaussian_if_under_val(1.01, (n_e, n_e), w_e_e, 0.3 * w_e_e),
		exp_if_under_val(1.01, (n_e, n_e), 0.1 * w_e_e)
	)

	w_initial[n_e:, :n_e] = gaussian_if_under_val(1.01, (n_i, n_e), w_e_i, 0.3 * w_e_i)
	w_initial[:n_e, n_e:] = gaussian_if_under_val(1.01, (n_e, n_i), w_i_e, 0.3 * np.abs(w_i_e))

	all_w_initial.append(w_initial)

# Defining L2 loss and objective function

r_target = np.zeros((len(t), n_e))
period = 6e-3
offset = 2e-3

for i in range(n_e):
	active_range = (period * i + offset, period * (i+1) + offset)
	n_t_steps = int(period / dt)
	t_step_start = int(active_range[0] / dt)
	r_target[t_step_start:(t_step_start + n_t_steps), i] = np.sin(np.pi/period * dt * np.arange(n_t_steps))

def l2_loss(r, r_target):
	if np.isnan(r).any():
		return 10000
	return np.sum(np.square(r[:, :n_e] - r_target))

eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
}

def simulate_single_network(w_initial, plasticity_coefs):
	w = copy(w_initial)

	for i in range(N_INNER_LOOP_ITERS):
		r, s, v, w_out = simulate_seq(t, n_e, n_i, r_in, transfer_e, transfer_i, plasticity_coefs, w, dt=dt, tau_e=5e-3, tau_i=0.1e-3, g=1, w_u=1)
		# dw_aggregate = np.sum(np.abs(w_out - w))
		if np.isnan(r).any():
			return r, w
		w = w_out

	return r, w

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None):
	start = time.time()

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs)
	results = pool.map(f, all_w_initial)
	pool.close()

	loss = np.sum([l2_loss(res[0], r_target) for res in results]) + 100 * np.sum(np.abs(plasticity_coefs))

	if eval_tracker is not None:
		scale = 1
		fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10 * scale, 3  * scale))

		r, w = results[-1]

		for l_idx in range(r.shape[1]):
			if l_idx < n_e:
				if l_idx % 1 == 0:
					axs[0, 0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)])
					axs[0, 0].plot(t, r_target[:, l_idx], '--', c=layer_colors[l_idx % len(layer_colors)])
			else:
				axs[0, 1].plot(t, r[:, l_idx], c='black')

		axs[1, 0].matshow(w_initial)
		axs[1, 1].matshow(w)

		axs[0, 0].set_title(f'Loss: {loss}')

		pad = 4 - len(str(eval_tracker['evals']))
		zero_padding = '0' * pad
		evals = eval_tracker['evals']

		fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')

		eval_tracker['best_loss'] = loss
		eval_tracker['evals'] += 1

		plt.close('all')

	dur = time.time() - start
	print('duration:', dur)
	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')

	return loss

simulate_plasticity_rules(np.zeros(81), eval_tracker=eval_tracker)

x0 = np.zeros(81)
options = {
	'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
}

x, es = cma.fmin2(partial(simulate_plasticity_rules, eval_tracker=eval_tracker), x0, 0.1, options=options)
print(x)
print(es.result_pretty())
