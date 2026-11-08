#!/bin/bash
# 极端日试验（云端，Python/HiGHS）：与基准集合同一口径——两步法 C&CG + 邻域搜索，再做 365 日样本外测试。
# 可续跑：已有结果的步骤自动跳过；容器重启后重新运行本脚本即可。
cd /home/user/yelendatacenter/v3/code
R=../results/ext2; mkdir -p $R
export HIGHS_THREADS=1 STEP1=mono GAP1=3e-3 TL1=900 GAP2=1e-3 TL2=1800 EPS=1e-3 PAR_LS=1 LS_MAX=30 PARAM_OVER='{"K": 16}'
job() {   # $1 = 标签（SP_S6 / DRO_S5 / RO_S7）
  [ -f $R/zhongwei_ext_$1.pkl ] || OUT=$R/zhongwei_ext_$1.pkl python -u ccg_solve.py zhongwei_ext S5 $1 ccg - >> $R/log_$1.txt 2>&1
  [ -f $R/zhongwei_ext_$1.pkl ] && { [ -f $R/zhongwei_ext_oos_$1.csv ] || SRC=zhongwei_ext RDIR=$R PAR=1 python -u oos_test.py zhongwei $1 >> $R/log_oos_$1.txt 2>&1; }
}
job SP_S6 & job DRO_S5 & job RO_S7 &
wait
echo "极端日试验完成 $(date)" >> $R/done.txt
