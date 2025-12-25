#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p {{ partition }}
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=130:00:00
#SBATCH --ntasks=30
#SBATCH --mem=200G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8

## Commands to run
source python-env/bin/activate

python train_integrator.py \
    --std_expl 0.001 \
    --pool_size 30 \
    --batch 10 \
    --fixed_data 1 \
    --l1_pen 5e-7 \
    --asp 50 \
    --frac_inputs_fixed 0 \
    --syn_change_prob 0 \
    --seed 2000 \
    --struct_prior hard_coded \
    --bump_init 1 \
    --train 0 \
    --exp_title "hardcoded_n5_w__1.0_inp_0.3_het_0" \
    --dc_input 0.2 \
    --inh_het 0 \
    --cell_type_1_size 5 \
    --HR_to_HD_width 1 \
    --HD_to_HR_width 0 \
    --input_size 3 \
    --time 1.0 \
    --time_test 1.0 \
    --time_input 0.27 \
    --bump_amp 0.05 \
    --bump_init_onset 0.01 \
    --p_active_floor 0.25 \
    --enable_diag \
    --w_e_e 1.02e-4 \ 

deactivate

## Exit
exit 0