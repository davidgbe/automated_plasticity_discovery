#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=5
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python ei_activity_shaping_test.py --pool_size 5 --batch 1 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0 --syn_change_prob 0 --stdp_tau STDP_TAU --fr_tau FR_TAU --stdp_coef STDP_COEF --w_ie W_IE --seed SEED --index INDEX

deactivate

## Exit
exit 0
