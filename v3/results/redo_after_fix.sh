#!/bin/bash
# 同时充放修复后重做：极端日样本外测试、典型日导出（S1/S3/S4/S5）、表F、物理检查
cd /home/user/yelendatacenter/v3/code
while pgrep -f fix_simul_results.py >/dev/null; do sleep 20; done
export HIGHS_THREADS=1
R=../results
mkdir -p $R/old_before_fix && mv $R/ext2/zhongwei_ext_oos_*.csv $R/old_before_fix/ 2>/dev/null
for t in SP_S6 DRO_S5 RO_S7; do SRC=zhongwei_ext RDIR=$R/ext2 PAR=2 python -u oos_test.py zhongwei $t >> $R/log_redo.txt 2>&1; done
python -u export_days.py zhongwei $R/zhongwei_DRO_S5.pkl >> $R/log_redo.txt 2>&1
for t in DRO_S1 DRO_S3 DRO_S4; do OUTDIR=$R/fig_data_$t python -u export_days.py zhongwei $R/zhongwei_$t.pkl 2023-11-22 >> $R/log_redo.txt 2>&1; done
mv $R/zhongwei_tableF.csv $R/old_before_fix/ 2>/dev/null; PAR=2 python -u table_f.py zhongwei >> $R/log_redo.txt 2>&1
python check_physics_v4.py >> $R/log_redo.txt 2>&1
echo "重做完成 $(date)" >> $R/log_redo.txt
