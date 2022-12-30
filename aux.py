# miscellaneous useful functions and classes
import numpy as np
import os
from copy import deepcopy as copy


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


def burst_count(ndarr):
	cnts_per_nrn = ndarr.sum(axis=0)
	return cnts_per_nrn, cnts_per_nrn.mean(), cnts_per_nrn.std()


def uncertainty_plot(ax, x, y, y_stds):
	ax.plot(x, y)
	ax.fill_between(x, y - y_stds, y + y_stds)


def bin_occurrences(occurrences, min_val=0, max_val=None, bin_size=1):
    scaled_occurrences = ((occurrences - min_val) / bin_size).astype(int)

    if max_val is None:
        max_val = occurrences.max()

    max_idx = int(np.ceil((max_val - min_val) / bin_size)) + 1

    binned = np.zeros(max_idx, dtype=int)
    for i, n in enumerate(scaled_occurrences):
        if n >= max_idx or n < 0:
            raise IndexError(f'val {occurrences[i]} is out of bounds for min {min_val} and max {max_val}')
        binned[n] += 1
    return np.arange(max_idx) * bin_size, binned


def calc_degree_dist(mat):
    degree_freqs = bin_occurrences(np.count_nonzero(mat, axis=1))
    return np.arange(len(degree_freqs)), degree_freqs


def rand_n_ones_in_vec_len_l(n, l):
    if n > l:
        raise ValueError('n cannot be greater than l')
    vec = np.concatenate([np.ones(n, int), np.zeros(l - n, int)])
    return vec[np.random.permutation(l)]


def rand_per_row_mat(n, shape):
    return np.stack([rand_n_ones_in_vec_len_l(n, shape[1]) for i in range(shape[0])])

def mat_1_if_under_val(val, shape):
    return np.where(np.random.rand(*shape) < val, 1, 0)

def gaussian_if_under_val(val, shape, mean, std):
    return np.where(np.random.rand(*shape) < val, np.random.normal(loc=mean, scale=std, size=shape), 0)

def exp_if_under_val(val, shape, scale):
    return np.where(np.random.rand(*shape) < val, np.random.exponential(scale=scale, size=shape), 0)

def dropout_on_mat(mat, percent, min_idx=0, max_idx=None):
    if max_idx is None:
        max_idx = mat.shape[1]

    num_idxs_in_bounds = max_idx - min_idx

    survival_indices = rand_n_ones_in_vec_len_l(int((1. - percent) * num_idxs_in_bounds), num_idxs_in_bounds)
    survival_indices = np.concatenate([np.ones(min_idx), survival_indices, np.ones(mat.shape[1] - max_idx)])

    m = copy(mat)
    m[:, survival_indices == 0] = 0
    return m, survival_indices

def rev_argsort(arr):
    arr_argsort = np.flip(np.argsort(arr))
    rev_argsorted = np.zeros(len(arr))
    for i, ind in enumerate(arr_argsort):
        rev_argsorted[ind] = i
    return rev_argsorted

def set_smallest_n_zero(arr_ref, n, arr_set=None):
    if arr_set is None:
        arr_set = copy(arr_ref)

    sort_indices = rev_argsort(np.abs(arr_ref))

    for i, sort_i in enumerate(sort_indices):
        if sort_i >= (len(arr_ref) - n):
            arr_set[i] = 0
    return arr_set
