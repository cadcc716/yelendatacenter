#!/bin/bash
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
run() { printf "%s\n" "$@" | xargs -P 4 -L 1 sh -c 'TL=$0 python3 exp.py plan $1' >> $L 2>&1; }
run "5400 Z-M2" "5400 Z-R100-SP" "5400 Z-R100-U3"
run "5400 Z-R100-U2"
until grep -q "^\[done\] Z-M3 " $L; do sleep 60; done
echo ZW_STAGE2 >> $L
printf "%s\n" Z-M3 Z-M2 Z-R100-SP Z-R100-U2 Z-R100-U3 | xargs -P 4 -L 1 python3 exp.py polish >> $L 2>&1
for n in Z-R100-SP Z-R100-U2 Z-R100-U3; do python3 exp.py fixout $n >> $L 2>&1; done
printf "%s\n" Z-M0 Z-M3 Z-R100-SP Z-R100-U3 | xargs -P 4 -L 1 python3 exp.py post >> $L 2>&1
python3 exp.py design >> $L 2>&1; python3 exp.py collect >> $L 2>&1
echo ZW_ALL_DONE >> $L
