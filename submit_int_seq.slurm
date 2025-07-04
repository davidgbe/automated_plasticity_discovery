#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --ntasks=30
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python train_integrator.py --std_expl 1e-3 --pool_size 30 --batch 10 --fixed_data 1 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015 --hd_hd_sparsity 0.2 --hd_hr_sparsity 0.2 --struct_prior seq --bump_init 1

deactivate

## Exit
exit 0
