#!/bin/bash

## Presets
#SBATCH --job-name=clean_up

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=3
#SBATCH --mem=10G

#SBATCH --chdir=.

rm -rf ./sims_outs/**/*.svg

## Exit
exit 0
