#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

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

FILE_NAMES="
int_long_sim_PRIOR_shift_10_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8015_2025-07-08_16:03:13.715325
int_long_sim_PRIOR_shift_10_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8016_2025-07-08_16:03:13.715323
int_long_sim_PRIOR_shift_10_TH_0_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8017_2025-07-10_09:47:23.329528
int_long_sim_PRIOR_shift_10_TH_0_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8018_2025-07-10_09:51:50.338358
int_long_sim_PRIOR_shift_10_TH_0_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8019_2025-07-10_09:51:50.363894
int_long_sim_PRIOR_shift_10_TH_0_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8020_2025-07-10_09:51:50.364009
"

for RFN in $FILE_NAMES; do
    python test_integrator_2.py --std_expl 0 --pool_size 10 --batch 1 --fixed_data 1 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 2000 --root_file_name $RFN --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior shift --bump_init 1
done

deactivate

## Exit
exit 0
