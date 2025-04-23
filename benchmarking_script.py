import numpy  as np
from numba import njit
import time

@njit
def test_run(w : np.ndarray, r : np.ndarray, t : int):
    for i in range(t):
        np.dot(w, r)

w = np.random.rand(200, 200)
r = np.random.rand(200)

s = time.time()
test_run(w, r, 300 * 1000)
end = time.time()
print(end - s)

