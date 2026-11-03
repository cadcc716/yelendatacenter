"""蓄冷价值直接校验：以 S3 最优容量为基础，枚举 (E_TES, K_MC)，固定容量逐典型日精确求运行成本，按 p0 加权得能源侧年化成本。"""
import os, sys, pickle, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HIGHS_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def one(a):
    x, i, over = a
    P = PR.params(**over); D = dict(np.load(f"{ROOT}/data/zhongwei_data.npz")); fl = BO.flags_of("S4")
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, w=1.0, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-4, tl=600); return obj

if __name__ == "__main__":
    base = pickle.load(open(sys.argv[1], "rb")); over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    P = PR.params(**over); D = dict(np.load(f"{ROOT}/data/zhongwei_data.npz")); a = dict(PR.annual_cost(P), N=0.0)
    grid = [tuple(map(float, g.split(":"))) for g in os.environ["GRID"].split(",")] if os.environ.get("GRID") else [(t, m) for t in (0, 5000, 10000, 20000) for m in (3048, 2700, 2400, 2100)]
    jobs = [(dict(base["x"], TES=float(t), MC=float(m)), i, over) for t, m in grid for i in base["days"]]
    with Pool(int(os.environ.get("PAR", 3))) as p: Q = np.array(p.map(one, jobs)).reshape(len(grid), -1)
    p0 = np.asarray(base["p0"])
    for (t, m), q in zip(grid, Q):
        x = dict(base["x"], TES=float(t), MC=float(m)); inv = sum(a[c] * x[c] for c in BO.CAPS)
        print(f"E_TES {t/1e3:5.1f} MWh, K_MC {m/1e3:5.3f} MW: 能源侧 {(inv + 365 * p0 @ q) / 1e4:8.2f} 万元 (投资 {inv/1e4:7.2f}, 运行 {365 * p0 @ q / 1e4:7.2f})", flush=True)
