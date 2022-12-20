#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks=11
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd /mmfs1/gscratch/labfairhall/davidgbe
source davidgbe-env/bin/activate
cd $CURRDIR

python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-4 1e-4 1e-4 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:42:57.261446
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-5 1e-5 1e-5 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:42:57.261446
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-6 1e-6 1e-6 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:42:57.261446

python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-4 1e-4 1e-4 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:44:23.590097
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-5 1e-5 1e-5 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:44:23.590097
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-6 1e-6 1e-6 --q 0.8 --load_initial seq_very_loose_rough_STD_EXPL_0.0025_FIXED_False_L1_PENALTY_1e-06_1e-06_1e-06_2022-12-20_00:44:23.590097

deactivate

## Exit
exit 0