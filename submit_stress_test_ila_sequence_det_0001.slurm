#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --ntasks=20
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0000 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0004 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0008 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0012 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0016 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0020 --seed 444 --root_file_name ila_train_det_input_0_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_8001_2024-05-19_18:28:39.418818 --title nonmult_bound --time 0.11
deactivate

## Exit
exit 0
