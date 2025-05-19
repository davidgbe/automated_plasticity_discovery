# miscellaneous useful functions and classes
import numpy as np
import os
from copy import deepcopy as copy
from time import time
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import re


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


@partial(jax.jit, static_argnames=['shape'])
def jax_gaussian_if_under_val(key, val, shape, mean=0, std=1):
    key1, key2 = jr.split(key)
    return jnp.where(
        jr.uniform(key1, shape) < val,
        jr.normal(key2, shape) * std + mean,
        0,
    )


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


def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str


def start_timer():
    start_time = time()
    def end_time():
        diff = time() - start_time
        print(f'Completed in {diff} seconds')
        sys.stdout.flush()
        return diff
    return end_time


def merge_with_indices_jax(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    merged = jnp.concatenate([a, b])
    sort_order = jnp.argsort(merged)
    sorted_merged = merged[sort_order]

    origin = jnp.concatenate([
        jnp.zeros(len(a), dtype=int),
        jnp.ones(len(b), dtype=int)
    ])
    origin_sorted = origin[sort_order]

    a_indices = jnp.nonzero(origin_sorted == 0, size=len(a))[0]
    b_indices = jnp.nonzero(origin_sorted == 1, size=len(b))[0]

    return sorted_merged, a_indices, b_indices


def find_dirs_with_fragment(base_path, frag):
    pattern = re.compile(frag)
    return [
        name for name in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, name)) and pattern.search(name)
    ]


def format_plot(
    axs,
    linewidth=1,
    ticklength=8,
    ticklabelsize=12,
    axislabelsize=13,
    tickwidth=1,
    rightspine=False,
    leftspine=True,
    topspine=False,
    bottomspine=True,
    ):

    if type(axs) is not list and type(axs) is not np.array and type(axs) is not np.ndarray:
        axs = [axs]

    if type(axs) is np.ndarray:
        axs = axs.flatten()

    for ax in axs:
        ax.spines['top'].set_visible(topspine)
        ax.spines['right'].set_visible(rightspine)
        ax.spines['bottom'].set_visible(bottomspine)
        ax.spines['left'].set_visible(leftspine)

        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['left'].set_linewidth(linewidth)

        ax.tick_params(axis='both', length=ticklength, labelsize=ticklabelsize, width=tickwidth)

        ax.xaxis.label.set_size(axislabelsize)
        ax.yaxis.label.set_size(axislabelsize)
