import os
from collections import OrderedDict
import functools


base_path = os.curdir
scripts = [
	'submit_test_seq.slurm',
]

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		if k == 'TITLE': 
			continue
		s = s.replace(k, v)

	if 'TITLE' in repl_dict:
		s = s.replace('TITLE', repl_dict['TITLE'])

	return s

def format_title(params):
	title = ''
	for k, v in params.items():
		title += ('_' + k + '_' + v)
	return title

def iter_range(r, n):
	if n == 1:
		yield (0, r[0])
	else:
		for i in range(n):
			yield (i, i * (r[1] - r[0]) / (n - 1) + r[0])

def map_to_list(func, l):
    '''
    Maps the list 'l' through the function 'func'
    Parameters
    ----------
    func : function
        Takes a single argument of type of 'l'
    l : list
    '''
    return list(map(func, l))

def reduce_mult(l):
    return functools.reduce(lambda e1, e2: e1 * e2, l, 1)

# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
    domain = map_to_list(lambda a: len(a), arrs)
    coordinate_lists = []
    for i, dim in enumerate(domain):
        coords = []
        mult = 1
        if i != len(domain) - 1:
            mult = reduce_mult(domain[i+1:])
        for e in arrs[i]:
            coords += (mult * [e])
        repeat_factor = reduce_mult(domain[0:i])
        if repeat_factor > 0:
            coords *= repeat_factor
        coordinate_lists.append(coords)
    return coordinate_lists

def pad_zeros(to_pad, length):
	padded = str(to_pad)
	while len(padded) < length:
		padded = '0' + padded
	return padded

batch_size = 1

params = OrderedDict()
params['SEED'] = [str(500)]
params['ROOT_FILE_NAME'] = [
	# TO TEST
    # # 'decoder_ei_rollback_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00072_FRACI_0.75_SEED_8000_2023-12-19_15:25:03.097712',
    # # 'decoder_ei_rollback_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00072_FRACI_0.75_SEED_8001_2023-12-19_20:38:37.781770',
    # 'decoder_ei_rollback_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00072_FRACI_0.75_SEED_8002_2023-12-19_20:38:37.868671',
    # 'decoder_ei_rollback_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00072_FRACI_0.75_SEED_8003_2023-12-19_20:41:15.347403',
    # # 'decoder_ei_rollback_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00072_FRACI_0.75_SEED_8004_2023-12-19_20:50:41.475679',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8005_2024-02-08_13:40:54.782819',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8006_2024-02-08_17:21:35.152834',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8007_2024-02-08_18:14:44.591173',
    # # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8008_2024-02-08_18:14:44.590143',
    # # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8009_2024-02-08_18:29:43.490886',
    # # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8010_2024-02-08_19:54:11.573400',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8011_2024-02-08_20:20:42.029971',
    # # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8012_2024-02-08_20:27:38.677022',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8013_2024-02-08_20:27:50.409418',
    # # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8014_2024-02-08_21:36:35.534398',
    # 'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8015_2024-02-15_13:46:11.581933',
    # 'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8016_2024-02-16_06:37:48.620849',
    # 'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8017_2024-02-16_07:02:27.408270',
    'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8018_2024-02-16_08:24:59.736918',
    'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8019_2024-02-16_12:42:11.335319',
    # 'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8020_2024-02-16_15:13:15.747325',
    # 'decoder_ei_mixed_extra_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8021_2024-02-16_15:35:06.710145',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8022_2024-02-21_18:16:43.240051',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8023_2024-02-21_15:34:30.720676',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8024_2024-02-21_19:13:01.531987',
    'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8025_2024-02-21_19:13:34.486542',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8026_2024-02-21_20:58:29.570289',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8027_2024-02-21_19:33:51.145795',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8028_2024-02-22_00:51:39.554855',
    # 'decoder_ei_mixed_long10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.00073_FRACI_0.75_SEED_8029_2024-02-21_23:56:24.339139',
]

n_seeds = len(params['SEED'])

for key in params.keys():
	if key == 'SEED' or type(params[key][0]) is str:
		continue
	params[key] = [str(v[1]) for v in iter_range(params[key][0], params[key][1])]

all_values = cartesian(*(params.values()))
n_scripts = len(all_values[0])
n_scripts_exec = 0

for src_name in scripts:
	for n in range(0, n_scripts, batch_size):

		name_parts = src_name.split('.')
		dst_name = name_parts[0] + '_' + pad_zeros(n, 4) + '.' + name_parts[1]

		src = open(src_name, 'rt')
		dst = open(dst_name, 'wt')

		for line in src:
			if line.find('python') >= 0:
				for batch_idx in range(batch_size):
					script_num = n + batch_idx
					if script_num >= n_scripts:
						continue

					augmented_params = {}

					for param_idx, v in enumerate(params.keys()):
						augmented_params[v] = all_values[param_idx][script_num]

					augmented_params['TITLE'] = format_title(augmented_params)
					augmented_params['INDEX'] = str(int(script_num / n_seeds))

					line_replaced = replace_all(line, augmented_params)
					dst.write(line_replaced)
			else:
				dst.write(line)
		src.close()
		dst.close()

		os.system('sbatch ./' + dst_name)
		print(dst_name)