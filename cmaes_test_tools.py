import os
import numpy as np
from functools import reduce, partial

def generate_surrogate_problem(dims, sparse_dims):
	extra_dims = dims - sparse_dims

	means_sparse = 0.5 * np.random.uniform(0.1, 1, size=sparse_dims)
	stds_sparse = 0.5 * np.random.uniform(0.1, 0.3, size=sparse_dims)
	means = np.concatenate([means_sparse, np.zeros(extra_dims)])
	stds = np.concatenate([stds_sparse, 0.1 * np.ones(extra_dims)])

	def f(X, eval_tracker=None):
		losses = []
		for x in X:
			if eval_tracker is not None:
				eval_tracker['evals'] += 1
			shifted_means = x - means
			loss = -np.exp(-0.5 * np.sum(np.power(shifted_means / stds, 2)))
			losses.append(loss)
		return losses, 0

	return f, means_sparse