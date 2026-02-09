#!/bin/bash

## Presets
#SBATCH --job-name=active_coding

#SBATCH -p cpu-g2
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --ntasks=20
#SBATCH --mem=50G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python sweep_active_tuning_params_2.py

deactivate

## Exit
exit 0
