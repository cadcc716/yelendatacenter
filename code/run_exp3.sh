#!/bin/bash
cd "$(dirname "$0")"
./run_exp2.sh
python3 exp.py design >> ../results/exp/log.txt 2>&1
python3 exp.py collect >> ../results/exp/log.txt 2>&1
echo ALL_DONE >> ../results/exp/log.txt
