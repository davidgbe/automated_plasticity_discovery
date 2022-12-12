#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=34:00:00
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

python train_sequence.py --std_expl 0.025 --mcp_t 1e-4 --mcp_s 0.1  --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.025 --mcp_t 1e-5 --mcp_s 0.1  --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.025 --mcp_t 1e-4 --mcp_s 0.05 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.025 --mcp_t 1e-5 --mcp_s 0.05 --dw_pen 0 --pool_size 10 --batch 10

deactivate

## Exit
exit 0