#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --ntasks=20
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python test_sequence_single_cell.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0.00072 --seed SEED --root_file_name ROOT_FILE_NAME

deactivate

## Exit
exit 0
