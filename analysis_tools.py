import os
import csv
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from csv_reader import read_csv
from copy import deepcopy as copy

def format_single_axs(axs):
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

def format_axs(axs):
    if type(axs) is list or type(axs) is np.ndarray:
        for i in range(len(axs)):
            format_single_axs(axs[i])
    else:
        format_single_axs(axs)

def calc_entropy(l):
    abs_l = np.abs(l)
    per_l = abs_l / np.sum(abs_l)
    return -np.dot(np.log(per_l), per_l)

def find_desc_indices(x, losses):
    min_loss = losses[0]
    x_mins = [0]

    for i, x_i in enumerate(x):
        if losses[i] < min_loss:
            x_mins.append(x_i)
            min_loss = losses[i]
    return np.array(x_mins)
