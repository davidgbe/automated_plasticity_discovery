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

python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 30 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --q 1
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 30 --fixed_data 1 --l1_pen 1e-7 1e-7 1e-7 --q 1
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 30 --fixed_data 1 --l1_pen 5e-8 5e-8 5e-8 --q 1

deactivate

## Exit
exit 0