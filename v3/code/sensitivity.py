"""
灵敏度分析（执行方案 5.6 节），均以 S5、两步法（单层 MILP(40)）+ 容量邻域搜索求解：
  alpha : α1 ∈ {0.2, 0.4, 0.5, 0.7, 0.9, 0.99}（C12、图13）
  M     : M ∈ {90, 180, 365, 730}（图13）
  heat  : 热价 0.10–0.30 元/kWh_th（C14、图16 的一维切片）
  chip  : 芯片温度上限 80–90 ℃（图15）
用法：python sensitivity.py <site> <组名 ...>        结果存 results/<site>_sens_<组>_<取值>.pkl
"""
import os, sys, pickle, subprocess, time, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = {
    "alpha": [("ALPHA1", v) for v in (0.2, 0.4, 0.7, 0.9, 0.99)],          # 0.5 为基准，取主算例结果
    "M": [("M", v) for v in (90, 180, 730)],
    "heat": [("C_HEAT", v) for v in (0.10, 0.15, 0.25, 0.30)],
    "chip": [("T_CHIP_MAX", v) for v in (80.0, 83.0, 87.0, 90.0)],
}
CODE = """
import sys, json, pickle, os
sys.path.insert(0, {code!r})
import ccg_solve as CG
over = json.loads({over!r})
r = CG.two_step({site!r}, "S5", "DRO_sens", "mono", None, over)
pickle.dump(r, open({out!r}, "wb"))
"""


if __name__ == "__main__":
    site = sys.argv[1]; groups = sys.argv[2:] or list(GROUPS)
    par = int(os.environ.get("PAR", 2)); env = dict(os.environ, HIGHS_THREADS="1", STEP1="mono", GAP1="3e-3", TL1="900", GAP2="1e-3", TL2="1500", PAR_LS="1", LS_MAX="30")
    queue = [(g, k, v) for g in groups for k, v in GROUPS[g]]; run = {}
    while queue or run:
        while queue and len(run) < par:
            g, k, v = queue.pop(0); out = f"{ROOT}/results/{site}_sens_{g}_{v}.pkl"
            if os.path.exists(out): continue                      # 已完成的点跳过
            code = CODE.format(code=os.path.dirname(os.path.abspath(__file__)), over=json.dumps({k: v}), site=site, out=out)
            log = open(f"{ROOT}/results/log_sens_{g}_{v}.txt", "w")
            run[(g, v)] = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=log, stderr=log, env=env)
        for key, p in list(run.items()):
            if p.poll() is not None: print(key, "完成", p.returncode, flush=True); run.pop(key)
        time.sleep(5)
