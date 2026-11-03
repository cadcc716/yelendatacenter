#!/bin/bash
# 试算：SP（θ=0）、单层 MILP、MIPGap=3e-3，两步法，S3→S4→S5 嵌套热启动
cd /home/user/yelendatacenter/v3/code
export HIGHS_THREADS=1 STEP1=mono GAP1=3e-3 GAP2=3e-3 TL1=1500 TL2=1500
T=../results/trial; TAG=${1:-r0}
OUT=$T/S1_$TAG.pkl python -u ccg_solve.py zhongwei S1 SP_S1 mono - > $T/log_S1_$TAG.txt 2>&1 &
OUT=$T/S3_$TAG.pkl python -u ccg_solve.py zhongwei S3 SP_S3 mono - > $T/log_S3_$TAG.txt 2>&1 &&
OUT=$T/S4_$TAG.pkl python -u ccg_solve.py zhongwei S4 SP_S4 mono $T/S3_$TAG.pkl > $T/log_S4_$TAG.txt 2>&1 &&
OUT=$T/S5_$TAG.pkl python -u ccg_solve.py zhongwei S5 SP_S5 mono $T/S4_$TAG.pkl > $T/log_S5_$TAG.txt 2>&1
wait
echo "trial $TAG done"
