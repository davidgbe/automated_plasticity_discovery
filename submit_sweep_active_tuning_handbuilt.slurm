#!/bin/bash

## Presets
#SBATCH --job-name=active_coding

#SBATCH -p compute
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=10
#SBATCH --mem=50G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python sweep_active_tuning_params.py

deactivate

## Exit
exit 0
