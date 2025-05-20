import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(matrix, ax=None, xlabel='Time', ylabel='Neuron index', title='Activity heatmap',
                 cmap='viridis', vmin=None, vmax=None, figsize=None, save_path=None):
    """
    Plots an N x T matrix as a heatmap into the provided axis.

    Parameters:
    - matrix: 2D NumPy array of shape (N, T)
    - ax: Matplotlib Axes object where the heatmap will be plotted
    - xlabel, ylabel, title: Labels for the plot
    - cmap: Colormap for the heatmap
    - vmin, vmax: Color scale limits (set to None for automatic scaling)
    """
    matrix = np.asarray(matrix)
    assert matrix.ndim == 2, "Input must be a 2D matrix (N x T)"

    if ax is None:
        kwargs = {}
        if figsize is not None:
            kwargs['figsize'] = figsize

        fig, ax = plt.subplots(1, 1, **kwargs)
    else:
        fig = None

    im = ax.imshow(matrix, aspect='auto', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activity')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if save_path is not None and fig is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        plt.close()

    return ax, fig


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

# def plot_results(results, eval_tracker, out_dir, plasticity_coefs, true_losses, syn_effect_penalties, total_activity_penalties, train=True):
# 	scale = 3
# 	n_res_to_show = BATCH_SIZE

# 	gs = gridspec.GridSpec(4 * n_res_to_show + 3, 2)
# 	fig = plt.figure(figsize=(4  * scale, (4 * n_res_to_show + 3) * scale), tight_layout=True)
# 	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(4 * n_res_to_show)]
# 	axs += [fig.add_subplot(gs[4 * n_res_to_show, :])]
# 	axs += [fig.add_subplot(gs[4 * n_res_to_show + 1, :])]
# 	axs += [fig.add_subplot(gs[4 * n_res_to_show + 2, :])]

# 	all_effects = []

# 	for i in np.arange(BATCH_SIZE):
# 		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
# 		res = results[i]
# 		r = res['r']
# 		r_exp_filtered = res['r_exp_filtered']
# 		w = res['w']
# 		w_initial = res['w_initial']
# 		effects = res['syn_effects']
# 		all_weight_deltas = res['all_weight_deltas']
# 		rs_for_loss = res['rs_for_loss']
# 		r_in_for_loss = res['r_in_for_loss']
# 		targets_for_loss = res['targets_for_loss']

# 		all_effects.append(effects)

# 		plotted_trial_count = 0

# 		for trial_idx in range(rs_for_loss.shape[0]):
# 			if trial_idx < rs_for_loss.shape[0] - 3:
# 				continue
# 			r = rs_for_loss[trial_idx, ...]
# 			r_in = r_in_for_loss[trial_idx, ...]
# 			targets = targets_for_loss[trial_idx, ...]

# 			# scale = 1
# 			# fig_r_in, axs_r_in = plt.subplots(1, 1, figsize=(4 * scale, 2 * scale), sharex=True, sharey=True)
# 			# axs_r_in.plot(np.arange(len(targets)), targets, color='red')
# 			# input_diffs = r_in[input_start:, n_e_pool + n_e_side : n_e_pool + 2 * n_e_side].sum(axis=1) - r_in[input_start:, n_e_pool:n_e_pool + n_e_side].sum(axis=1)
# 			# input_summed = [input_diffs[:j].sum() for j in range(len(input_diffs))]
# 			# axs_r_in.plot(np.arange(len(targets)), input_summed, color='black')

# 			# fig_r_in.savefig(f'{out_dir}/r_in_trial_{trial_idx}.png')

# 			for l_idx in range(r.shape[1]):
# 				if l_idx < n_e_pool:
# 					pass
# 					# if l_idx % 1 == 0:
# 					# 	axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
# 				elif l_idx >= (r.shape[1] - n_i):
# 					axs[4 * i + plotted_trial_count][1].plot(t, r[:, l_idx], c='black') # graph inh activity

# 			axs[4 * i + plotted_trial_count][0].matshow(r[:, :n_e_pool + 2 * n_e_side].T, aspect=1/0.1)
# 			plotted_trial_count += 1

# 		r_exc = r[:, :n_e_pool]
# 		r_summed = np.sum(r_exc, axis=0)
# 		r_active_mask =  np.where(r_summed != 0, 1, 0).astype(bool)
# 		r_summed_safe_divide = np.where(r_active_mask, r_summed, 1)
# 		r_normed = r_exc / r_summed_safe_divide
# 		t_means = np.sum(t.reshape(t.shape[0], 1) * r_normed, axis=0)
# 		# t_ordering = np.argsort(t_means)
# 		# t_ordering = np.concatenate([t_ordering, np.arange(n_e, n_e + n_i)])

# 		# sorted_w_initial = w_initial[t_ordering, :][:, t_ordering]
# 		# sorted_w = w[t_ordering, :][:, t_ordering]

# 		vmin = np.min([w_initial.min(), w.min()])
# 		vmax = np.max([w_initial.max(), w.max()])

# 		vbound = np.max(w)

# 		mappable = axs[4 * i + 3][0].matshow(w_initial, vmin=-vbound, vmax=vbound, cmap='bwr') # plot initial weight matrix
# 		plt.colorbar(mappable, ax=axs[4 * i + 3][0])

# 		mappable = axs[4 * i + 3][1].matshow(w, vmin=-vbound, vmax=vbound, cmap='bwr') # plot final weight matrix
# 		plt.colorbar(mappable, ax=axs[4 * i + 3][1])

# 		axs[4 * i][0].set_title(f'{true_losses[i]} + {syn_effect_penalties[i]} + {total_activity_penalties[i]}')
# 		for i_axs in range(2):
# 			axs[2 * i][i_axs].set_xlabel('Time (s)')
# 			axs[2 * i][i_axs].set_ylabel('Firing rate')

# 		axs[4 * n_res_to_show + 2].plot(np.arange(len(all_weight_deltas)), np.log(all_weight_deltas), label=f'{i}')

# 	partial_rules_len = int(len(plasticity_coefs))

# 	all_effects = np.array(all_effects)
# 	effects = np.mean(all_effects, axis=0)

# 	axs[4 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
# 	effects_argsort = []
# 	for l in range(1):
# 		effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
# 		effects_argsort_partial = np.flip(np.argsort(effects_partial))
# 		effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
# 		x = np.arange(len(effects_argsort_partial)) + l * partial_rules_len
# 		axs[4 * n_res_to_show + 1].bar(x, effects_partial[effects_argsort_partial], zorder=0)
# 		for i_e in x:
# 			axs[4 * n_res_to_show + 1].scatter(i_e * np.ones(all_effects.shape[0]), all_effects[:, effects_argsort_partial][:, i_e], c='black', zorder=1, s=3)
# 	axs[4 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
# 	axs[4 * n_res_to_show + 1].set_xlim(-1, len(effects))

# 	true_loss = np.sum(true_losses)
# 	syn_effect_penalty = np.sum(syn_effect_penalties)
# 	total_activity_penalty = np.sum(total_activity_penalties)
# 	axs[4 * n_res_to_show].set_title(f'Loss: {true_loss + syn_effect_penalty}, {true_loss}, {syn_effect_penalty}, {total_activity_penalty}')

# 	# plot the coefficients assigned to each plasticity rule (unsorted by size)
# 	for l in range(1):
# 		axs[4 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
# 	axs[4 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
# 	axs[4 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
# 	axs[4 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))

# 	axs[4 * n_res_to_show + 2].set_xlabel('Epochs')
# 	axs[4 * n_res_to_show + 2].set_ylabel('log(delta W)')
# 	axs[4 * n_res_to_show + 2].legend()

# 	pad = 4 - len(str(eval_tracker['evals']))
# 	zero_padding = '0' * pad
# 	evals = eval_tracker['evals']

# 	# fig.tight_layout()
# 	if train:
# 		fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
# 	else:
# 		fig.savefig(f'{out_dir}/{zero_padding}{evals}_test.png')
# 	plt.close('all')