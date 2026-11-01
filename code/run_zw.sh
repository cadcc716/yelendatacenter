#!/bin/bash
# 审稿第 1 项: 宁夏中卫 ERA5 多年逐时气象 (2014–2024, 按年份划分) + BurstGPT 188 日负荷; 需网络放行 archive-api.open-meteo.com
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
python3 -c "import exp; exp.dataset('zw')" >> $L 2>&1 || { echo "ZW_DATA_FAILED" >> $L; exit 1; }
run() { printf "%s\n" "$@" | xargs -P 4 -L 1 sh -c 'TL=$0 python3 exp.py plan $1' >> $L 2>&1; }
run "600 Z-M0" "600 Z-M0-设计工况" "5400 Z-M3" "5400 Z-M1"
run "5400 Z-M2" "5400 Z-R100-SP" "5400 Z-R100-U2" "5400 Z-R100-U3"
for n in Z-R100-SP Z-R100-U2 Z-R100-U3; do python3 exp.py fixout $n >> $L 2>&1; python3 exp.py post $n >> $L 2>&1; done
python3 exp.py design >> $L 2>&1; python3 exp.py collect >> $L 2>&1
echo ZW_ALL_DONE >> $L
