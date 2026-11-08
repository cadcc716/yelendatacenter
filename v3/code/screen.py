"""筛选试算：固定 GPU 数量 N，按给定参数快速求解 S5（单层 MILP，第二步），导出 6 月 15 日与冬季典型日，计算 C10/C11/C6 等指标。
用法：python screen.py <名称> <N> '<PARAM_OVER json>' [场景=S5]"""
import os, sys, json, subprocess, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{ROOT}/results/screen"; os.makedirs(R, exist_ok=True)
name, N, over = sys.argv[1], sys.argv[2], sys.argv[3]; scen = sys.argv[4] if len(sys.argv) > 4 else "S5"
out = f"{R}/{name}_{scen}.pkl"
env = {**os.environ, "HIGHS_THREADS": "1", "FIXN": N, "SKIP1": "1", "LS": "0", "GAP2": "5e-3", "TL2": os.environ.get("TL2", "600"),
       "EPS": "5e-3", "PARAM_OVER": over, "OUT": out}
if not os.path.exists(out):
    subprocess.run([sys.executable, "-u", "ccg_solve.py", "zhongwei", scen, f"DRO_{name}", "mono", "-"], env=env,
                   stdout=open(f"{R}/{name}_{scen}.log", "w"), stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)))
if scen == "S5":
    od = f"{R}/{name}_days"
    if not os.path.exists(f"{od}/zhongwei_2023-11-22_逐时.csv"):
        subprocess.run([sys.executable, "export_days.py", "zhongwei", out, "2023-11-22"], env={**os.environ, "OUTDIR": od},
                       stdout=open(f"{R}/{name}_export.log", "w"), stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)))
