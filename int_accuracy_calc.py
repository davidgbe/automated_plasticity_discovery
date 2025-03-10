import numpy as np

n_blocks = 16
n_trials = 1000

inputs = np.where(np.random.rand(n_trials, n_blocks) > 0.5, 1, -1)
s = np.sum(inputs, axis=1)
m = np.mean(inputs[s > 0, :].sum(axis=1))
guesses = np.where(s > 0, m, -m)

print('s', s)
print('m', guesses)

print(np.sum(np.square(s - guesses)) / np.sum(np.square(s)))
