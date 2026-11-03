"""3(a)：固定 S5 当前容量，单独求解 2023-06-15，逐项放松候选约束，定位实时任务缺额的约束来源。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X5 = dict(N=4279, PV=15000, WT=0, PB=6525, EB=23716, FC=7302, MC=2296, TES=15570, HP=1420)   # S5 第1轮容量
CASES = [("基准", {}, {}), ("冷机容量放开(K_MC=10 MW)", {"MC": 10000}, {}), ("取消式(12)第二行(时段末温区)", {}, {"GATE_NEXT": False}),
         ("并网上限放开(30 MW)", {}, {"P_GRID_MAX": 30000.0}), ("GPU数量放开(N=6000)", {"N": 6000}, {}),
         ("液冷热容×2", {}, {"C_W": 2300.0}), ("自然冷却容量放开(K_FC=12 MW)", {"FC": 12000}, {}), ("蓄冷容量放开(60 MWh)", {"TES": 60000}, {})]

def run(c):
    name, dx, dp = c
    P = PR.params(**dp); D = dict(np.load(f"{ROOT}/data/zhongwei_data.npz")); i = list(D["dates"]).index("2023-06-15")
    fl = BO.flags_of("S5"); M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix={**X5, **dx})
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-4, tl=900); R = BO.extract(sol, V)
    return name, R, obj, st, D, i, P

if __name__ == "__main__":
    os.environ["HIGHS_THREADS"] = "1"
    with Pool(4) as p: res = p.map(run, CASES)
    name, R0, _, _, D, i, P = res[0]
    print("气温:", np.round(D["Tamb"][i]).astype(int))
    print("实时到达:", np.round(D["AR"][i]).astype(int))
    rows = {}
    for name, R, obj, st, *_ in res:
        rows[name] = np.round(R["LR"]).astype(int)
        print(f"{name:24s} 状态 {st}, 日成本 {obj/1e4:6.2f} 万元, 实时缺额 {R['LR'].sum():6.0f} GPU·h ({int((R['LR']>1).sum())} h)")
    T = pd.DataFrame(rows).T; T.columns = [f"{h}时" for h in range(24)]
    print(T.loc[:, (T > 0).any()].to_string())
    R = R0; m = np.zeros((5, 24))
    for a, k in enumerate(P["K_R"]): m[k] += R["mR"][a].sum(0)
    for a, k in enumerate(P["K_B"]): m[k] += R["mB"][a].sum(0)
    print("基准 温区:", R["u"].argmax(0) + 1)
    print("基准 T_w:", np.round(R["Tw"][:24], 1))
    print("基准 工作GPU:", np.round(m.sum(0)).astype(int))
    for k in range(5): print(f"基准 档位{k+1}:", np.round(m[k]).astype(int))
    print("基准 q_MC:", np.round(R["qMC"].sum(0)).astype(int))
    print("基准 q_FC:", np.round(R["qFC"].sum(0)).astype(int))
    print("基准 放冷:", np.round(R["qdis"]).astype(int))
    print("基准 购电:", np.round(R["buy"]).astype(int))
    print("基准 βP_IT:", np.round(R["PIT"] * P["BETA"]).astype(int))
