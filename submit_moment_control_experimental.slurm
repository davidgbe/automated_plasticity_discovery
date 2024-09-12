#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=5
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load foster/python/miniconda/3.8 
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
source python-env/bin/activate

python moment_control_experimental.py --pool_size 5 --batch 1 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob SYN_CHANGE_PROB --exc_rule_setpoint EXC_RULE_SETPOINT --inh_rule_setpoint INH_RULE_SETPOINT --exc_rule_rate EXC_RULE_RATE --inh_rule_rate INH_RULE_RATE --exc_m1_c EXC_M1_C --exc_m2_c EXC_M2_C --inh_m1_c INH_M1_C --inh_m2_c INH_M2_C --a_i A_I --T_i T_I --a_f A_F --T_f T_F --title TITLE
 

deactivate

## Exit
exit 0
