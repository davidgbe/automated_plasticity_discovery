#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p cpu-g2-mem2x
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --ntasks=30
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source ../miniconda3/etc/profile.d/conda.sh
conda activate cont_attractor

python train_integrator.py --std_expl 1e-3 --pool_size 30 --batch 10 --fixed_data 1 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015 --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior one_in_one_out --bump_init 1

deactivate

## Exit
exit 0
