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
    --std_expl {{ std_expl }} \
    --pool_size 30 \
    --batch 10 \
    --fixed_data 1 \
    --l1_pen 5e-7 \
    --asp 50 \
    --frac_inputs_fixed 0 \
    --syn_change_prob 0 \
    --seed {{ seed }} \
    --struct_prior shift \
    --bump_init 1 \
    --train 0 \
    --exp_title "{{ title }}_{{ time }}_inp_{{ time_input }}_het_{{ inh_het }}" \
    --dc_input 0.2 \
    --inh_het {{ inh_het }} \
    --cell_type_1_size 5 \
    --HR_to_HD_width 1 \
    --HD_to_HR_width 0 \
    --input_size 3 \
    --time {{ time }} \
    --time_test {{ time }} \
    --time_input {{ time_input }} \
    --bump_amp {{ bump_amp }} \
    --bump_init_onset {{ bump_init_onset }} \
    --p_active_floor {{ p_active_floor }} \
    --enable_diag \
    --root_file_name {{ root_file_name }} \
    --save_all_w

deactivate

## Exit
exit 0