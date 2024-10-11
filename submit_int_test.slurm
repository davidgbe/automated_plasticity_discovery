#!/bin/bash

## Presets
#SBATCH --job-name=int_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=20
#SBATCH --mem=40G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

RFN=etc.
python test_integrator.py --std_expl 0.0005 --pool_size 5 --batch 1 --fixed_data 1 --l1_pen 5e-7 --asp 500 --frac_inputs_fixed 0 --syn_change_prob 0 --seed 2000 --root_file_name $RFN

deactivate

## Exit
exit 0
