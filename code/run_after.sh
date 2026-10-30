#!/bin/bash
# 等当前批次结束后运行训练样本量实验, 然后设计日校核与汇总
cd "$(dirname "$0")"
until grep -q ALL_DONE ../results/exp/log.txt; do sleep 60; done
export HIGHS_THREADS=1
printf "plan %s\n" N30-M3 N30-U2 N30-U3 N60-M3 N60-U2 N60-U3 | TL=3600 xargs -P 4 -L 1 python3 exp.py >> ../results/exp/log.txt 2>&1
python3 exp.py design >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_DONE2 >> ../results/exp/log.txt
