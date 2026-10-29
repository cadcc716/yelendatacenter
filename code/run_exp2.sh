#!/bin/bash
# 第二轮: 主要方案延长时限重解 (TL 秒), 然后其余作业; 每行 "TL=<秒> <exp.py 参数>"
cd "$(dirname "$0")"
export HIGHS_THREADS=1
NJ=${NJ:-4}
run() { tl=${1#TL=}; shift; if [ "$1" = "ops" ]; then until [ -f ../results/exp/row_M3.pkl ] && ! pgrep -f "exp.py plan M3$" >/dev/null; do sleep 30; done; fi; TL=$tl python3 exp.py "$@"; }
export -f run
xargs -P "$NJ" -L 1 bash -c 'run "$@"' _ < ../results/exp/jobs2.txt >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_FINISHED >> ../results/exp/log.txt
