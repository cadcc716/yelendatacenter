#!/bin/bash
# E3 补充: 风险成本型 (软约束) 规划 SP / U1 / U2 / U3
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
printf "%s\n" R100-SP R100-U1 R100-U2 R100-U3 | TL=5400 xargs -P 4 -L 1 python3 exp.py plan >> $L 2>&1
printf "%s\n" R100-SP R100-U1 R100-U2 R100-U3 | xargs -P 4 -L 1 python3 exp.py polish >> $L 2>&1
printf "%s\n" R100-SP R100-U1 R100-U2 R100-U3 | xargs -P 4 -L 1 python3 exp.py post >> $L 2>&1
echo SOFT_BASE_DONE >> $L
printf "%s\n" R30-SP R30-U2 R30-U3 R300-SP R300-U2 R300-U3 | TL=2400 xargs -P 4 -L 1 python3 exp.py plan >> $L 2>&1
printf "%s\n" R30-SP R30-U2 R30-U3 R300-SP R300-U2 R300-U3 | xargs -P 4 -L 1 python3 exp.py polish >> $L 2>&1
python3 exp.py design >> $L 2>&1; python3 exp.py collect >> $L 2>&1
echo SOFT_ALL_DONE >> $L
