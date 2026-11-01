#!/bin/bash
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
until grep -q SOFT_ALL_DONE $L; do sleep 60; done
printf "%s\n" R100-SP R100-U1 R100-U2 R100-U3 | xargs -P 4 -L 1 python3 exp.py polish >> $L 2>&1
printf "%s\n" R100-SP R100-U1 R100-U2 R100-U3 | xargs -P 4 -L 1 python3 exp.py post >> $L 2>&1
for n in R30-SP R30-U2 R30-U3 R300-SP R300-U2 R300-U3 R100-SP R100-U1 R100-U2 R100-U3; do python3 exp.py fixout $n >> $L 2>&1; done
python3 exp.py design >> $L 2>&1; python3 exp.py collect >> $L 2>&1; python3 risk_e.py >> $L 2>&1
echo SOFT2_DONE >> $L
