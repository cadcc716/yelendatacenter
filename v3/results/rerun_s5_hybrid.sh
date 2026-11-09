#!/bin/bash
cd /home/user/yelendatacenter/v3/code
while pgrep -f "improve_cross.py DRO_S5" >/dev/null; do sleep 20; done
HIGHS_THREADS=1 python -u improve_cross.py DRO_S5 20 > ../results/log_improve_DRO_S5_hybrid.txt 2>&1
