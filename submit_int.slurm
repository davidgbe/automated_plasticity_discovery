#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p ckpt-all
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task 4
#SBATCH --time=80:00:00
#SBATCH --mem=50G

#SBATCH --chdir=.

## Module import
CURRDIR=$(pwd)
echo $CURRDIR

module load cuda/12.6.3
source ../miniconda3/etc/profile.d/conda.sh
conda activate cont_attractor

python train_integrator.py --std_expl 1e-3 --batch 10 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015 --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior random

## Exit
exit 0
