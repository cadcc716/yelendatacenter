#!/bin/bash
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
until grep -q REV_ALL_DONE $L; do sleep 60; done
python3 - > /tmp/polish_list.txt <<'PY'
import pickle, glob, os
for f in sorted(glob.glob("../results/exp/plan_*.pkl")):
    n = os.path.basename(f)[5:-4]
    if n.startswith(("SENS-", "FAN-", "L-")) and not n.startswith("L-R100") and n.endswith(("M2", "M3")):
        if pickle.load(open(f, "rb"))["gap"] > 3e4: print(n)
PY
xargs -P 4 -L 1 python3 exp.py polish < /tmp/polish_list.txt >> $L 2>&1
python3 exp.py collect >> $L 2>&1; python3 rob_e.py >> $L 2>&1
echo REV2_DONE >> $L
