#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p ckpt-all
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --ntasks=20
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source ../miniconda3/etc/profile.d/conda.sh
conda activate cont_attractor

RFN="int_preexist_inputs_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8051_2025-03-31_14:10:57.049743"
RFN="int_preexist_inputs_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8052_2025-03-31_22:17:51.916099"
RFN="int_preexist_inputs_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8053_2025-03-31_22:34:23.654335"
RFN="int_preexist_inputs_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8054_2025-03-31_22:35:39.288195"

RFN="int_preexist_inputs_asym_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8051_2025-04-03_18:56:20.335393"
RFN="int_preexist_inputs_asym_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8052_2025-04-03_18:56:20.335369"
RFN="int_preexist_inputs_asym_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8053_2025-04-03_22:10:24.675697"
RFN="int_preexist_inputs_asym_rand10_STD_EXPL_0.001_FIXED_True_L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_1.0_FRACI_0.0_SEED_8054_2025-04-04_06:18:06.402567"

python test_integrator_2.py --std_expl 0 --pool_size 10 --batch 1 --fixed_data 1 --l1_pen 5e-7 --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 2000 --root_file_name $RFN --hd_hd_sparsity 0.8

deactivate

## Exit
exit 0
