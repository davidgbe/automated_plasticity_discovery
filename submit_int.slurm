#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --ntasks=20
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python train_integrator.py --std_expl 0.003 --pool_size 20 --batch 5 --fixed_data 1 --l1_pen 1e-8 1e-8 1e-8 --asp 500 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015

deactivate

## Exit
exit 0
