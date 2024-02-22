#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=15
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python refit_sequence.py --std_expl 0.003 --pool_size 15 --batch 10 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0.75 --syn_change_prob 0.00073 --load_initial ROOT_FILE_NAME --seed SEED

deactivate

## Exit
exit 0
