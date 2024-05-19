#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=20
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0 --seed 444 --root_file_name ila_rules_artificial_growth_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.00072_FRACI_0.75_SEED_5001_2024-05-18_23:08:23.768743 --title stress_test_ila_preconfig
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.00072 --seed 444 --root_file_name ila_rules_artificial_growth_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.00072_FRACI_0.75_SEED_5001_2024-05-18_23:08:23.768743 --title stress_test_ila_preconfig
deactivate

## Exit
exit 0
