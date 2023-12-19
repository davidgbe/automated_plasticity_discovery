#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --ntasks=30
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python train_integrator.py --std_expl 0.003 --pool_size 30 --batch 3 --fixed_data 1 --l1_pen 1e-8 1e-8 1e-8 --asp 1 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015

deactivate

## Exit
exit 0
