#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

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

python ei_activity_shaping_test_2.py --pool_size 5 --batch 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0 --syn_change_prob SYN_CHANGE_PROB --stdp_tau STDP_TAU --fr_tau FR_TAU --stdp_coef STDP_COEF --w_ie W_IE --seed SEED --index INDEX --w_ie_coef W_IE_COEF --y2_ie_coef Y2_IE_COEF

deactivate

## Exit
exit 0
