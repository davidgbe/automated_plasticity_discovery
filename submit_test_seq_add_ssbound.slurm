#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=10
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python test_sequence.py --pool_size 10 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0 --seed 2222 --root_file_name kernel_test_small_net_1_BATCH_10_FIXED_True_L1_PENALTY_5e-07_5e-07_5e-07_CHANGEP_0.0_FRACI_0.75_SEED_555_2024-04-30_14:45:05.409985 --title add_ssbound --time 0.2
deactivate

## Exit
exit 0
