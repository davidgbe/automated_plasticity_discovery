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

python train_sequence.py --std_expl 0.015 --mcp_t 2e-4 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.015 --mcp_t 4e-4 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.015 --mcp_t 6e-4 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.015 --mcp_t 8e-4 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.015 --mcp_t 1e-3 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10
python train_sequence.py --std_expl 0.015 --mcp_t 2e-3 --mcp_s 0.3 --dw_pen 0 --pool_size 10 --batch 10

deactivate

## Exit
exit 0