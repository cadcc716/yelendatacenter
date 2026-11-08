#!/bin/bash
# 蓄冷单价 80 → 60 后重算含蓄冷的场景 S4–S7（S1–S3 不含蓄冷，结果不变，沿用）
cd /home/user/yelendatacenter/v3/code
R=../results; S=zhongwei
export HIGHS_THREADS=1 STEP1=mono GAP1=3e-3 TL1=900 GAP2=1e-3 TL2=1800 EPS=1e-3 PAR_LS=2 LS_MAX=50
ccg()  { python -u ccg_solve.py $S $1 $2 ccg  ${3:--} > $R/log_${S}_$2_ccg.txt  2>&1; }
mono() { LS=0 python -u ccg_solve.py $S $1 $2 mono ${3:--} > $R/log_${S}_$2_mono.txt 2>&1; }

( ccg S4 DRO_S4 $R/${S}_DRO_S3.pkl && ccg S5 DRO_S5 $R/${S}_DRO_S4.pkl; echo "C&CG S4→S5 完成" ) &
( ccg S5 SP_S6; echo "C&CG S6 完成" ) &
( ccg S5 RO_S7; echo "C&CG S7 完成" ) &
( mono S4 DRO_S4 $R/${S}_DRO_S3_mono.pkl && mono S5 DRO_S5 $R/${S}_DRO_S4_mono.pkl && mono S5 SP_S6 && mono S5 RO_S7; echo "单层 MILP 校验完成" ) &
wait
echo "第二批完成"
