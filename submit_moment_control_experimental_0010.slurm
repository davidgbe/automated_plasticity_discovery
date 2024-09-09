#!/bin/bash

## Presets
#SBATCH --job-name=seq_cmaes

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

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

python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.005 --a_f 0.6 --T_f 0.005 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.005_A_F_0.6_T_F_0.005
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.005 --a_f 0.6 --T_f 0.007 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.005_A_F_0.6_T_F_0.007
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.007 --a_f 0.5 --T_f 0.005 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.007_A_F_0.5_T_F_0.005
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.007 --a_f 0.5 --T_f 0.007 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.007_A_F_0.5_T_F_0.007
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.007 --a_f 0.6 --T_f 0.005 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.007_A_F_0.6_T_F_0.005
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.1 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.6 --T_i 0.007 --a_f 0.6 --T_f 0.007 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.1_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.6_T_I_0.007_A_F_0.6_T_F_0.007
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.25 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.5 --T_i 0.005 --a_f 0.5 --T_f 0.005 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.25_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.5_T_I_0.005_A_F_0.5_T_F_0.005
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.25 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.5 --T_i 0.005 --a_f 0.5 --T_f 0.007 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.25_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.5_T_I_0.005_A_F_0.5_T_F_0.007
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.25 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.5 --T_i 0.005 --a_f 0.6 --T_f 0.005 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.25_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.5_T_I_0.005_A_F_0.6_T_F_0.005
python moment_control_experimental.py --pool_size 5 --batch 5 --fixed_data 1 --l1_pen 0 0 0 --frac_inputs_fixed 0 --syn_change_prob 0 --exc_rule_setpoint 0.1 --inh_rule_setpoint 0.4 --exc_rule_rate 0.25 --inh_rule_rate -0.03 --exc_m1_c 0 --exc_m2_c 1 --inh_m1_c 0 --inh_m2_c 0 --a_i 0.5 --T_i 0.005 --a_f 0.6 --T_f 0.007 --title _SEED_2_SYN_CHANGE_PROB_0_EXC_RULE_SETPOINT_0.1_EXC_RULE_RATE_0.25_INH_RULE_SETPOINT_0.4_INH_RULE_RATE_-0.03_EXC_M1_C_0_EXC_M2_C_1_INH_M1_C_0_INH_M2_C_0_A_I_0.5_T_I_0.005_A_F_0.6_T_F_0.007
 

deactivate

## Exit
exit 0
