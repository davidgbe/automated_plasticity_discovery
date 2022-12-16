#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks=11
#SBATCH --mem=10G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd /mmfs1/gscratch/labfairhall/davidgbe
source davidgbe-env/bin/activate
cd $CURRDIR

python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-9 1e-9 1e-9
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 5e-10 5e-10 5e-10
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 2.5e-10 2.5e-10 2.5e-10
python train_sequence.py --std_expl 0.0025 --pool_size 10 --batch 10 --fixed_data 0 --l1_pen 1e-10 1e-10 1e-10

deactivate

## Exit
exit 0