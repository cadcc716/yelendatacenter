#!/usr/bin/env bash
# 并行运行全部模型场景与敏感性算例 (每个作业单线程 HiGHS), 最后汇总
cd "$(dirname "$0")"
export HIGHS_THREADS=1
NJ=${NJ:-4}
python3 data_prep.py > ../results/data_prep_log.txt 2>&1
{ python3 -c "import model; print('\n'.join(model.VARIANTS))" | sed 's/^/job /'
  python3 -c "import run_all as R; print('\n'.join(f'sjob {n} {v} {m}' for n, v, m in R.SENS_CASES))"; } > ../results/jobs.txt
xargs -P "$NJ" -L 1 python3 run_all.py < ../results/jobs.txt >> ../results/run_log.txt 2>&1
python3 run_all.py collect >> ../results/run_log.txt 2>&1
