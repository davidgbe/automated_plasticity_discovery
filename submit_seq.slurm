#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks=32
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python train_sequence.py --std_expl 0.005 --pool_size 30 --batch 30 --fixed_data 1 --l1_pen 5e-7 5e-7 5e-7 --asp 1 -seed 2000

deactivate

## Exit
exit 0
