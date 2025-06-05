#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=20
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python test_sequence_2.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0 --seed 500 --root_file_name decoder_ee_rollback_rescaled_b_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.0_FRACI_0.75_SEED_8014_2023-12-06_04:47:17.612728

# python test_sequence_2.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0.00034 --seed 500 --root_file_name decoder_ee_rollback_rescaled_b_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.0_FRACI_0.75_SEED_8014_2023-12-06_04:47:17.612728
# python test_sequence_2.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0.00070 --seed 500 --root_file_name decoder_ee_rollback_rescaled_b_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.0_FRACI_0.75_SEED_8014_2023-12-06_04:47:17.612728
# python test_sequence_2.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0.00110 --seed 500 --root_file_name decoder_ee_rollback_rescaled_b_10_STD_EXPL_0.003_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_ACT_PEN_1_CHANGEP_0.0_FRACI_0.75_SEED_8014_2023-12-06_04:47:17.612728

deactivate

## Exit
exit 0
