"""
C&CG 求解两阶段分布鲁棒容量规划 式(32)–(39)
  主问题(35)：容量 x + 全部 K 个典型日运行变量 + 概率割 η ≥ 365·Σ_s p_s^(l) Q_s
  子问题(36)：固定 x，逐典型日求解运行 MILP，得 Q_s(x)
  子问题(37)：在综合范数模糊集内求最恶劣分布 p*（线性规划）
  θ1 = θ∞ = 0 → SP；θ1 = 2、θ∞ = 1 → RO
"""
import time, pickle, sys
import numpy as np
from scipy.optimize import linprog
import params as PR, build_op as BO
from milp import MILP, INF

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worst_p(Q, p0, th1, thi):
    """式(37)：max Σ p_s Q_s，s.t. Σp = 1，Σ|p − p0| ≤ θ1，|p_s − p0_s| ≤ θ∞。变量 [p, p+, p−]。"""
    K = len(Q); I = np.eye(K); Z = np.zeros((K, K))
    c = np.r_[-np.asarray(Q, float), np.zeros(2 * K)]
    Aeq = np.vstack([np.r_[np.ones(K), np.zeros(2 * K)], np.hstack([I, -I, I])])
    beq = np.r_[1.0, p0]
    Aub = np.vstack([np.r_[np.zeros(K), np.ones(2 * K)], np.hstack([Z, I, I])])
    bub = np.r_[th1, np.full(K, thi)]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (3 * K), method="highs")
    return np.clip(r.x[:K], 0, None) / r.x[:K].clip(0).sum()


def build_master(P, D, flags, days, cuts, x_lb=None):
    M = MILP(); X, a = BO.add_capacity(M, P, flags)
    if x_lb:
        for k, v in x_lb.items(): M.lb[list(BO.CAPS).index(k)][:] = v   # 仅用于调试
    Td = BO.tamb_design(D, P); Vs = []
    for i in days:
        Vs.append(BO.add_day(M, P, X, BO.day_of(D, i), flags, w=0.0, Tamb_des=Td))
    eta = M.var(1, -INF, INF, cost=1.0)
    for p in cuts:
        M.row([(eta, 1.0)] + [(V["Q"], -P["DAYS"] * p[s]) for s, V in enumerate(Vs)], lo=0.0)
    return M, X, Vs, eta, a


def solve_sub(P, D, flags, x, days, tl=600, gap=1e-4):
    Td = BO.tamb_design(D, P); Q = []; R = []
    for i in days:
        M = MILP(); X, _ = BO.add_capacity(M, P, flags, fix=x)
        V = BO.add_day(M, P, X, BO.day_of(D, i), flags, w=1.0, Tamb_des=Td)
        sol, st, obj, g = M.solve(gap=gap, tl=tl)
        if sol is None: raise RuntimeError(f"子问题不可行: 日 {i}, 状态 {st}")
        Q.append(obj); R.append(BO.extract(sol, V))
    return np.array(Q), R


def ccg(P, D, scen="S5", th=None, eps=1e-3, max_it=10, tl_master=2700, gap_master=2e-4, log=print, start=None):
    flags = BO.flags_of(scen); days = list(D["med"]); p0 = D["p0"]
    th1, thi = th if th is not None else PR.theta(P)
    cuts = [p0.copy()]; LB, UB = -np.inf, np.inf; hist = []; best = None; t0 = time.time(); x0 = None
    for k in range(1, max_it + 1):
        M, X, Vs, eta, a = build_master(P, D, flags, days, cuts)
        sol, st, obj, gabs = M.solve(gap=gap_master, tl=tl_master, start=x0)
        if sol is None: raise RuntimeError(f"主问题失败: {st}")
        x = {c: float(sol[X[c]][0]) for c in BO.CAPS}; x["N"] = round(x["N"])
        inv = sum(a[c] * x[c] for c in BO.CAPS)
        LB = max(LB, obj - gabs)
        Q, R = solve_sub(P, D, flags, x, days)
        p = worst_p(Q, p0, th1, thi) if (th1 > 0 or thi > 0) else p0.copy()
        ub_k = inv + P["DAYS"] * float(p @ Q)
        if ub_k < UB: UB = ub_k; best = dict(x=x, inv=inv, Q=Q, p=p, R=R, a=a)
        gap = (UB - LB) / abs(UB)
        hist.append(dict(it=k, LB=LB, UB=UB, gap=gap, t=time.time() - t0, master_status=st, master_gap=gabs / abs(obj)))
        log(f"[{scen} θ=({th1:.3g},{thi:.3g})] 迭代 {k}: LB {LB / 1e4:.1f} 万元, UB {UB / 1e4:.1f} 万元, gap {gap:.2e}, "
            f"N {x['N']}, FC {x['FC']:.0f}, MC {x['MC']:.0f}, TES {x['TES']:.0f}, HP {x['HP']:.0f}, PV {x['PV']:.0f}, WT {x['WT']:.0f}, "
            f"EB {x['EB']:.0f}, PB {x['PB']:.0f}, 用时 {time.time() - t0:.0f} s")
        if gap <= eps: break
        if any(np.allclose(p, c, atol=1e-6) for c in cuts): break   # 最恶劣分布已在割集中，下界无法再提高
        cuts.append(p)
    best.update(hist=hist, scen=scen, theta=(th1, thi), days=days, p0=p0)
    return best


if __name__ == "__main__":
    site = sys.argv[1]; scen = sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else scen
    th = {"SP": (0.0, 0.0), "RO": (2.0, 1.0)}.get(tag.split("_")[0], None)
    P = PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    r = ccg(P, D, scen, th=th)
    pickle.dump(r, open(f"{ROOT}/results/{site}_{tag}.pkl", "wb"))
