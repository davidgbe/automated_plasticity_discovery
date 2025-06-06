from jax import numpy as jnp
import jax
import matplotlib.pyplot as plt

ALPHA = 1000
BETA = 1/ALPHA
SOFTPLUS_TRANSITION = 1e-3

@jax.jit
def _softplus(a):
    return jnp.where(
        a > SOFTPLUS_TRANSITION,
        a,
        1/ALPHA * jnp.log1p(jnp.exp(a/BETA)),
    )

scale = 4
fig, axs = plt.subplots(1, 1, figsize=(1 * scale, 1 * scale))

x = jnp.linspace(-1, 1, 1000)
axs.plot(x, _softplus(x))

fig.show()

