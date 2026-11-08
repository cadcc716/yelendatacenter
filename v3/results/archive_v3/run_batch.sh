#!/bin/bash
# 全批次（中卫）：两步法 —— 第一步单层 MILP 求 N*，第二步 C&CG（能源侧成本）；嵌套热启动 S3→S4→S5、S1→S2；
# 第二步后做容量邻域搜索。单层 MILP(40) 校验链同样嵌套，不做邻域搜索，用于与 C&CG 第二步结果比较。
cd /home/user/yelendatacenter/v3/code
R=../results; S=zhongwei
export HIGHS_THREADS=1 STEP1=mono GAP1=3e-3 TL1=900 GAP2=1e-3 TL2=1800 EPS=1e-3 PAR_LS=2 LS_MAX=40
ccg()  { python -u ccg_solve.py $S $1 $2 ccg  ${3:--} > $R/log_${S}_$2_ccg.txt  2>&1; }
mono() { LS=0 python -u ccg_solve.py $S $1 $2 mono ${3:--} > $R/log_${S}_$2_mono.txt 2>&1; }

( ccg S3 DRO_S3 && ccg S4 DRO_S4 $R/${S}_DRO_S3.pkl && ccg S5 DRO_S5 $R/${S}_DRO_S4.pkl; echo "C&CG 链 S3→S5 完成" ) &
( ccg S1 DRO_S1 && ccg S2 DRO_S2 $R/${S}_DRO_S1.pkl && ccg S5 SP_S6 && echo "C&CG S1、S2、S6 完成" ) &
( ccg S5 RO_S7; echo "C&CG S7 完成" ) &
( mono S1 DRO_S1 && mono S2 DRO_S2 $R/${S}_DRO_S1_mono.pkl && mono S3 DRO_S3 && mono S4 DRO_S4 $R/${S}_DRO_S3_mono.pkl && \
  mono S5 DRO_S5 $R/${S}_DRO_S4_mono.pkl && mono S5 SP_S6 && mono S5 RO_S7; echo "单层 MILP 校验完成" ) &
wait
echo "全批次完成"
