#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p ckpt-all
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
    --seed 8001 \
    --struct_prior shift \
    --bump_init 1 \
    --train 1 \
    --exp_title "n5_long_min_pactive_0p25_1.0_inp_0.27_het_0" \
    --dc_input 0.2 \
    --run_num 000681 \
    --inh_het 0 \
    --cell_type_1_size 5 \
    --HR_to_HD_width 1 \
    --HD_to_HR_width 0 \
    --input_size 3 \
    --time 1.0 \
    --time_test 1.0 \
    --time_input 0.27 \
    --bump_amp 0.15 \
    --bump_init_onset 0.01 \
    --p_active_floor 0.25

deactivate

## Exit
exit 0