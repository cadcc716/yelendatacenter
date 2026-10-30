#!/bin/bash
cd "$(dirname "$0")"
until grep -q ALL_DONE3 ../results/exp/log.txt; do sleep 60; done
export HIGHS_THREADS=1
printf "polish %s\n" S2-坦帕-M3 S2-芝加哥-M3 S1-aN0.6-M3 S1-aN1.4-M3 | xargs -P 4 -L 1 python3 exp.py >> ../results/exp/log.txt 2>&1
python3 exp.py design >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_DONE4 >> ../results/exp/log.txt
