#!/bin/bash
# 训练样本量实验结束后, 以 90 min、2 线程重解未收敛的气候算例, 然后重新设计日校核与汇总
cd "$(dirname "$0")"
until grep -q ALL_DONE2 ../results/exp/log.txt; do sleep 60; done
export HIGHS_THREADS=2
printf "plan %s\n" S2-坦帕-M3 S2-芝加哥-M3 | TL=5400 xargs -P 2 -L 1 python3 exp.py >> ../results/exp/log.txt 2>&1
python3 exp.py design >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_DONE3 >> ../results/exp/log.txt
