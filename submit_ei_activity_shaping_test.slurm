#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --ntasks=30
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python ei_activity_shaping_test.py --std_expl 0.003 --pool_size 20 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 --frac_inputs_fixed 0 --syn_change_prob 0 --seed SEED --w_ie W_IE --w_ee W_EE

deactivate

## Exit
exit 0
