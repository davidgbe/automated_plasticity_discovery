import numpy as np
import cma
import pickle
import sys

from run_active_tuning_handbuilt_v2_n_greater_than_2 import train_multiple_networks

# ----------------------------
# Deterministic R² metric
# ----------------------------

def compute_r2(result, n, dt, input_start=1.5):
    input_start_int = int(input_start / dt)

    ending_locs = []
    int_values = []
    positions = np.arange(n)

    for epoch_data in result['last_20_epochs']:
        x_history = epoch_data['x_history']
        color_val = epoch_data['color_val']

        x_pool = x_history[:, :n]
        loc_trace = np.sum((x_pool + np.random.normal(size=n, scale=0.05)) * positions, axis=1) / (
            x_pool.sum(axis=1) + 1e-6
        )

        ending_loc = np.nanmean(loc_trace[input_start_int:])
        ending_locs.append(ending_loc)
        int_values.append(color_val)

    ending_locs = np.array(ending_locs)
    int_values = np.array(int_values)

    if np.std(ending_locs) < 1e-8 or np.std(int_values) < 1e-8:
        return 0.0

    r = np.corrcoef(ending_locs, int_values)[0, 1]
    r = np.where(np.isnan(r), 0, r)
    return r ** 2


# ----------------------------
# Objective function
# ----------------------------

def objective(theta):
    """
    theta = [
        learning_rate,
        alpha,
        alpha_outer_scale,
        hebbian_dx_scale,
        hebbian_dx_scale_conj,
        homeo_rate,
        presyn_setpoint,
        tau_x_trace_1,
        tau_z_trace_1,
        tau_x_trace_2,
        tau_z_trace_2,
        tau_x_trace_3,
        tau_x_trace_4,
        outer_zx_x_scale,
        outer_xtilde_x_scale,
        outer_x_xtilde_scale,
    ]
    """

    (
        learning_rate,
        alpha,
        alpha_outer_scale,
        hebbian_dx_scale,
        hebbian_dx_scale_conj,
        homeo_rate,
        presyn_setpoint,
        tau_x_trace_1,
        tau_z_trace_1,
        tau_x_trace_2,
        tau_z_trace_2,
        tau_x_trace_3,
        tau_x_trace_4,
        outer_zx_x_scale,
        outer_xtilde_x_scale,
        outer_x_xtilde_scale,
    ) = theta

    # Hard safety clamp
    if learning_rate <= 0 or alpha <= 0 or hebbian_dx_scale <= 0:
        print('happens')
        return 1e6

    try:
        results, _ = train_multiple_networks(
            n_networks=1,
            n_epochs=10000,
            n=5,
            t_sim=(0, 2.0),
            dt=1e-4,
            learning_rate=learning_rate,
            alpha=alpha,
            alpha_outer_scale=alpha_outer_scale,
            hebbian_dx_scale=hebbian_dx_scale,
            hebbian_dx_scale_conj=hebbian_dx_scale_conj,
            homeo_rate=homeo_rate,
            presyn_setpoint=presyn_setpoint,
            tau_x_trace_1=tau_x_trace_1 * 1e-3,
            tau_z_trace_1=tau_z_trace_1 * 1e-3,
            tau_x_trace_2=tau_x_trace_2 * 1e-3,
            tau_z_trace_2=tau_z_trace_2 * 1e-3,
            tau_x_trace_3=tau_x_trace_3 * 1e-3,
            tau_x_trace_4=tau_x_trace_4 * 1e-3,
            outer_zx_x_scale=outer_zx_x_scale,
            outer_xtilde_x_scale=outer_xtilde_x_scale,
            outer_x_xtilde_scale=outer_x_xtilde_scale,
            w_e_scale=2,
            w_pool_to_shift=0.5,
            w_shift_to_pool=0.3,
            weight_perturbation=1.0,
            peak_amp=0.5,
            seed=0,
        )

        r2 = compute_r2(results[0], n=5, dt=1e-4)

        if not np.isfinite(r2):
            return 1e6

        return -r2

    except Exception as e:
        return 1e6


# ----------------------------
# Run CMA-ES
# ----------------------------

if __name__ == "__main__":

    x0 = [
        10,    # learning_rate
        20.0,  # alpha
        1.0,   # alpha_outer_scale
        10.0,  # hebbian_dx_scale
        1.0,   # hebbian_dx_scale_conj
        0.1,   # homeo_rate
        12,    # presyn_setpoint
        20,    # tau_x_trace_1
        20,    # tau_z_trace_1
        20,    # tau_x_trace_2
        20,    # tau_z_trace_2
        20,    # tau_x_trace_3
        20,    # tau_x_trace_4
        0.0,   # outer_zx_x_scale
        0.0,   # outer_xtilde_x_scale
        0.0,   # outer_x_xtilde_scale
    ]

    sigma0 = 10.0

    lower_bounds = [0.001, 0.1, -5.0, -1000, -1000, 0,    8,   1,  1,  1,  1,  1,  1, -1000, -1000, -1000]
    upper_bounds = [25.0,  100.0, 5.0,  1000,  1000, 10, 100,  30, 30, 30, 30, 30, 30,  1000,  1000,  1000]

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "bounds": [lower_bounds, upper_bounds],
            "verb_disp": 1,
        },
    )

    best_val = np.inf
    while not es.stop():
        solutions = es.ask()
        values = [objective(s) for s in solutions]
        for i_val, val in enumerate(values):
            if val < best_val:
                print(solutions[i_val])
                print('new_best_val', val)
                best_val = val
        es.tell(solutions, values)
        es.disp()
        sys.stdout.flush()

    result = es.result

    best_params = {
        "learning_rate":        result.xbest[0],
        "alpha":                result.xbest[1],
        "alpha_outer_scale":    result.xbest[2],
        "hebbian_dx_scale":     result.xbest[3],
        "hebbian_dx_scale_conj":result.xbest[4],
        "homeo_rate":           result.xbest[5],
        "presyn_setpoint":      result.xbest[6],
        "tau_x_trace_1":        result.xbest[7],
        "tau_z_trace_1":        result.xbest[8],
        "tau_x_trace_2":        result.xbest[9],
        "tau_z_trace_2":        result.xbest[10],
        "tau_x_trace_3":        result.xbest[11],
        "tau_x_trace_4":        result.xbest[12],
        "outer_zx_x_scale":     result.xbest[13],
        "outer_xtilde_x_scale": result.xbest[14],
        "outer_x_xtilde_scale": result.xbest[15],
        "r2": -result.fbest,
    }

    print("\nBest parameters found:")
    print(best_params)

    with open("cmaes_plasticity_results_linear_v2.pkl", "wb") as f:
        pickle.dump(best_params, f)