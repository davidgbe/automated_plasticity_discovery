import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 1e-4  # time step
tau_rules = 5e-3  # time constant
t_max = 0.1  # total simulation time (100 ms)
n_steps = int(t_max / dt)

# Initial conditions
r_exp = 0.0  # initial value of r_exp

# Arrays to store results
time = np.zeros(n_steps)
r_history = np.zeros(n_steps)
r_exp_history = np.zeros(n_steps)

# Pulse parameters
pulse_start = 0.01  # start at 10 ms
pulse_duration = 0.0002  # 1 ms pulse
pulse_end = pulse_start + pulse_duration
pulse_amplitude = 1.0

# Time evolution
for i in range(n_steps):
    t = i * dt
    time[i] = t
    
    # Define r as a pulse
    if pulse_start <= t < pulse_end:
        r = pulse_amplitude
    else:
        r = 0.0
    
    r_history[i] = r
    r_exp_history[i] = r_exp
    
    # Update r_exp using the given equation
    delta_r_exp = (r - r_exp) * dt / tau_rules
    r_exp = r_exp + delta_r_exp

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time * 1000, r_history, 'r-', linewidth=2, label='r (input pulse)', alpha=0.7)
plt.plot(time * 1000, r_exp_history, 'b-', linewidth=2, label='r_exp (filtered)')
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Evolution of r_exp in response to 1ms pulse', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('./r_exp_simulation.png', dpi=150)
plt.close()

# Print some key information
print(f"Simulation parameters:")
print(f"  dt = {dt}")
print(f"  tau_rules = {tau_rules}")
print(f"  Total time = {t_max} s ({t_max*1000} ms)")
print(f"  Number of steps = {n_steps}")
print(f"\nPulse parameters:")
print(f"  Pulse starts at {pulse_start*1000} ms")
print(f"  Pulse duration = {pulse_duration*1000} ms")
print(f"  Pulse amplitude = {pulse_amplitude}")
print(f"\nInitial r_exp = {r_exp_history[0]:.6f}")
print(f"Peak r_exp = {np.max(r_exp_history):.6f}")
print(f"Final r_exp = {r_exp_history[-1]:.6f}")
print(f"\nTime constant tau = {tau_rules*1000} ms")