import cma
import os
import numpy as np
from functools import reduce, partial

extra_dims = 10
sparse_dims = 2
total_dims = sparse_dims + extra_dims

f_min_means_sparse = [1, 0.1]
f_min_stds_sparse = [0.5, 0.05]

f_min_means = np.concatenate([f_min_means_sparse, np.zeros(extra_dims)])
f_min_stds = np.concatenate([f_min_stds_sparse, 0.005 * np.ones(extra_dims)])

prefactor = -1 / (np.power((2 * np.pi), total_dims/2) * reduce(lambda a, b: a*b, f_min_stds, 1))

# print('MIN:', prefactor)

def f(x, eval_tracker=None):
	if eval_tracker is not None:
		eval_tracker['evals'] += 1
	shifted_means = x - f_min_means
	return prefactor * np.exp(-0.5 * np.sum(np.power(shifted_means / f_min_stds, 2)))

sigmas = [1, 0.1, 0.01, 0.001, 1e-4]

for s in sigmas:
	sigma_tally = []
	evals_tally = []

	for i in range(50):

		options = {
			'verb_filenameprefix': os.path.join('./cmaes_test', 'outcmaes/'),
			'verb_disp': 0,
			'maxfevals': 5000,
		}

		eval_tracker = {
			'evals': 0
		}

		x, es = cma.fmin2(
			partial(f, eval_tracker=eval_tracker),
			np.zeros(total_dims),
			1,
			restarts=10,
			bipop=True,
			options=options)

		# print(es)
		# print(x, f(x) / prefactor)

		print(i, f(x) / prefactor, eval_tracker['evals'])

		sigma_tally.append(f(x) / prefactor)
		evals_tally.append(eval_tracker['evals'])

	print('Final for:', s, 'Value:', np.mean(sigma_tally), np.std(sigma_tally))
	print('evals', np.mean(evals_tally), np.std(evals_tally))