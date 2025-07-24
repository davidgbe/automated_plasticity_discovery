#!/bin/bash

## Presets
#SBATCH --job-name=int_hard_coded

#SBATCH -p cpu-g2-mem2x
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=30
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

# ALL_W_EE="
#     0.90e-4
#     0.91e-4
#     0.92e-4
#     0.93e-4
#     0.94e-4
#     0.95e-4
#     0.96e-4
#     0.97e-4
#     0.98e-4
#     0.99e-4
#     1.00e-4
#     1.01e-4
#     1.02e-4
#     1.03e-4
#     1.04e-4
#     1.05e-4
# "

ALL_W_EE="
    1.06e-4
    1.07e-4
    1.08e-4
    1.09e-4
    1.10e-4
    1.11e-4
    1.12e-4
    1.13e-4
    1.14e-4
    1.15e-4
"



for W_EE in $ALL_W_EE; do
    python train_integrator.py --w_e_e $W_EE -std_expl 1e-3 --pool_size 30 --batch 10 --fixed_data 1 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 8015 --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior hard_coded --bump_init 1 --train 0 --exp_title hardcoded_instant_inh_${W_EE} --self_org_iters 0  --dc_input 0.2 --instant_inhibition 1
done

deactivate

## Exit
exit 0
