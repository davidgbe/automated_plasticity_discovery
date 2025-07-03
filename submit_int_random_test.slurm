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
module load foster/python/miniconda/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

FILE_NAMES="
int_n15_PRIOR_random_10_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8015_2025-06-16_19:02:11.332169
int_n15_PRIOR_random_10_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8016_2025-06-16_19:03:15.840835
int_n15_PRIOR_random_10_STD_EXPL_0.001__L1_PENALTY_5e-07_ACT_PEN_50.0_CHANGEP_0.0_SEED_8017_2025-06-16_19:04:24.406017
"
for RFN in $FILE_NAMES; do
    python train_integrator.py --std_expl 1e-3 --pool_size 10 --batch 1 --fixed_data 1 --l1_pen 5e-7  --asp 50 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 2000 --root_file_name $RFN --hd_hd_sparsity 1.0 --hd_hr_sparsity 1.0 --struct_prior random --bump_init 0
done

deactivate

## Exitis c
exit 0
