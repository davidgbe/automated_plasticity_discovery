#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p ckpt-all
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task 10
#SBATCH --time=01:00:00
#SBATCH --mem=50G

#SBATCH --chdir=.

## Module import
CURRDIR=$(pwd)
echo $CURRDIR

module load cuda/12.6.3
source ../miniconda3/etc/profile.d/conda.sh
conda activate cont_attractor

python test_hard_coded_integrator.py --std_expl 1e-3 --eta 10 --batch 10 --l1_pen 5e-7  --activity_penalty 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 9000 --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior shift --run_num 001020

## Exit
exit 0
