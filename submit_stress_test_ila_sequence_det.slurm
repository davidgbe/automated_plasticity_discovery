#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

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

python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0 --seed SEED --root_file_name ROOT_FILE_NAME --title nonmult_bound --time 0.11
python stress_test_ila_sequence_det.py --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.00072 --seed SEED --root_file_name ROOT_FILE_NAME --title nonmult_bound --time 0.11
deactivate

## Exit
exit 0
