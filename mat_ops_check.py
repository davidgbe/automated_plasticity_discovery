import numpy as np
from numba import njit
import numba
from time import time

n_total = 20
r_0 = np.ones(n_total)
r_1 = np.random.rand(n_total)
r_bin = (r_1 > 0.5).astype(float)

# r_bin_r_bin_true = np.outer(r_bin, r_bin)
# r_1_r_bin_true = np.outer(r_bin, r_1)


# r_bin_r_bin = np.empty((n_total, n_total), dtype=float)
# r_1_r_bin = np.empty((n_total, n_total), dtype=float)

# for r_index in range(n_total):
# 	r_bin_i = r_bin[r_index]
# 	if r_bin_i:
# 		r_bin_r_bin[r_index, :] = r_bin
# 		r_1_r_bin[r_index, :] = r_1
# 	else:
# 		r_bin_r_bin[r_index, :] = 0
# 		r_1_r_bin[r_index, :] = 0

# print((r_bin_r_bin == r_bin_r_bin_true).all())
# print((r_1_r_bin == r_1_r_bin_true).all())

# pop_sizes = [5, 3]

# r_1_pow_split = [np.random.rand(pop_sizes[0]), np.random.rand(pop_sizes[1])]
# r_exp_filtered_curr_split = [np.random.rand(15, pop_sizes[0]), np.random.rand(15, pop_sizes[1])]
# r_bin_split = [(r_1_pow_split[0] > 0.5).astype(float), (r_1_pow_split[1] > 0.5).astype(float)]

# p_j = 1
# p_i = 0
# k = 0

# r_bin_r_exp = np.empty((pop_sizes[p_j], pop_sizes[p_i]), dtype=float)

# for r_index in range(r_bin_r_exp.shape[1]):
# 	if r_bin_split[p_i][r_index]:
# 		r_bin_r_exp[:, r_index] = r_exp_filtered_curr_split[p_j][k + 3, :]
# 	else:
# 		r_bin_r_exp[:, r_index] = 0

# r_bin_r_exp_true = np.outer(r_exp_filtered_curr_split[p_j][k + 3, :], r_bin_split[p_i])

# print((r_bin_r_exp_true  == r_bin_r_exp).all())

# r_exp_r_bin = np.empty((pop_sizes[p_j], pop_sizes[p_i]), dtype=float)
# for r_index in range(r_exp_r_bin.shape[0]):
# 	if r_bin_split[p_j][r_index]:
# 		r_exp_r_bin[r_index, :] = r_exp_filtered_curr_split[p_i][k + 3, :]
# 	else:
# 		r_exp_r_bin[r_index, :] = 0

# print(r_exp_r_bin)

# r_exp_r_bin_true = np.outer(r_bin_split[p_j], r_exp_filtered_curr_split[p_i][k + 3, :])

# print(r_exp_r_bin_true)

# print((r_exp_r_bin_true  == r_exp_r_bin).all())

def outer_bin_vec_vec(bin_vec : np.ndarray, vec : np.ndarray):
	outer = np.zeros((vec.shape[0], bin_vec.shape[0]), dtype=np.float64)
	for index in range(outer.shape[1]):
		if bin_vec[index] > 0:
			outer[:, index] = vec
	return outer

n_total = 1000

start = time()

for i in range(100):
	r_0 = np.ones(n_total)
	r_1 = np.random.rand(n_total)
	r_bin = (r_1 > 0.5).astype(float)

	outer_bin_vec_vec(r_bin, r_1)

print(time() - start)

start = time()

for i in range(100):
	r_0 = np.ones(n_total)
	r_1 = np.random.rand(n_total)
	r_bin = (r_1 > 0.5).astype(float)

	print(np.outer(r_bin, r_1))

print(time() - start)

print((np.outer(r_1, r_bin) == outer_bin_vec_vec(r_bin, r_1)).all())



