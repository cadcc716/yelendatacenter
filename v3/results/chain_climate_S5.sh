#!/bin/bash
cd /home/user/yelendatacenter/v3/code
while pgrep -f "improve_cross.py DRO_S4 15 guian" >/dev/null; do sleep 30; done
HIGHS_THREADS=1 python -u improve_cross.py DRO_S5 15 guian > ../results/log_improve_guian_S5.txt 2>&1
while pgrep -f "ccg_solve.py shaoguan S5" >/dev/null || pgrep -f "improve_cross.py DRO_S4 15 shaoguan" >/dev/null; do sleep 30; done
HIGHS_THREADS=1 python -u improve_cross.py DRO_S5 15 shaoguan > ../results/log_improve_shaoguan_S5.txt 2>&1
echo done > ../results/climate_improve_done.txt
