#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --ntasks=20
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

# python train_ila_sequence.py --std_expl 0.33 --pool_size 6 --batch 2 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0 --param_vec PARAM_VEC --seed SEED --title TITLE

python train_ila_sequence.py --std_expl 0.33 --pool_size 20 --batch 10 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --frac_inputs_fixed 0.75 --syn_change_prob 0.0 --param_vec "-0.001" 0.001 0 "-0.02" 0.02 0 2.5 7.5 0.01 10e-3 10e-3 2e-3 10e-3 10e-3 2e-3 0 0 0 1 1 0 1 1 1 --seed 5000 --title ila_rules_artificial_growth
deactivate

## Exit
exit 0