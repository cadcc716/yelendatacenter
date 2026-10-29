#!/bin/bash
# 修订版 E1–E5 并行驱动: NJ 个作业并行, 每个 HiGHS 单线程; ops 等待 M3 规划完成
cd "$(dirname "$0")"
export HIGHS_THREADS=1
NJ=${NJ:-4}
run() { if [ "$1" = "ops" ]; then until [ -f ../results/exp/row_M3.pkl ]; do sleep 30; done; fi; python3 exp.py "$@"; }
export -f run
xargs -P "$NJ" -L 1 bash -c 'run "$@"' _ < ../results/exp/jobs.txt >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_FINISHED >> ../results/exp/log.txt
