#!/bin/bash
# 审稿前三项: 负荷 188 日 (L-), 参数稳健性 (SENS-), 风机变工况 (FAN-)
cd "$(dirname "$0")"; export HIGHS_THREADS=1; L=../results/exp/log.txt
run() { printf "%s\n" "$@" | xargs -P 4 -L 1 sh -c 'TL=$0 python3 exp.py plan $1' >> $L 2>&1; }
run "600 SENS-Rth0.8-M0" "600 SENS-Rth1.2-M0" "600 SENS-Tc83-M0" "600 SENS-Tc87-M0" "600 SENS-Poff0.08-M0" "600 SENS-Poff0.15-M0" \
    "600 SENS-rho0.1-M0" "600 SENS-rho0.4-M0" "600 FAN-M0" "600 FAN-M0-设计工况" "600 L-M0" "600 L-M0-设计工况"
echo REV_STAGE1 >> $L
run "3600 L-M3" "3600 FAN-M2" "2700 FAN-M3" "2700 FAN-M1" "1800 SENS-Rth0.8-M3" "1800 SENS-Rth1.2-M3" "1800 SENS-Tc83-M3" \
    "1800 SENS-Tc87-M3" "1800 SENS-Poff0.08-M3" "1800 SENS-Poff0.15-M3" "1800 SENS-rho0.1-M3" "1800 SENS-rho0.4-M3"
echo REV_STAGE2 >> $L
run "3600 L-R100-SP" "3600 L-R100-U3"
for n in L-R100-SP L-R100-U3; do python3 exp.py fixout $n >> $L 2>&1; python3 exp.py post $n >> $L 2>&1; done
python3 exp.py design >> $L 2>&1; python3 exp.py collect >> $L 2>&1
echo REV_ALL_DONE >> $L
