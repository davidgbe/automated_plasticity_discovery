#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=5
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python test_sequence.py --pool_size 5 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0 --seed 2222 --root_file_name kernel_test_final_160_0_mult_ssbound_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_603_2024-05-10_15:37:53.641905 --title nonmult_bound --time 0.11
deactivate

## Exit
exit 0
