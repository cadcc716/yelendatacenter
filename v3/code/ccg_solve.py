"""
两阶段分布鲁棒容量规划 式(32)–(39)：C&CG 求解与等价单层 MILP
  C&CG 主问题(35)：容量 x（N 取连续变量）+ 全部 K 个典型日运行变量 + 概率割；上一轮解热启动；LB 取 best bound
  子问题(36)：固定 x，逐典型日求解运行 MILP；子问题(37)：最恶劣分布线性规划
  收敛后 N 取整，固定全部容量重算运行成本与最恶劣分布（最终报告值）
  等价单层 MILP：对(37)取对偶，
      min C_inv(x) + λ + Σ p0_s π_s + θ1 μ + θ∞ Σ ν_s
      s.t. λ + π_s ≥ 365·C_op,s(y_s)，−(μ+ν_s) ≤ π_s ≤ μ+ν_s，μ, ν_s ≥ 0，y_s ∈ Y(x, ξ_s)
  θ1 = θ∞ = 0 → SP；θ1 = 2、θ∞ = 1 → RO
用法：python ccg_solve.py <site> <开关场景S1–S5> <结果标签> [ccg|mono]
"""
import os, sys, time, pickle, json
import numpy as np
from scipy.optimize import linprog
import params as PR, build_op as BO
from milp import MILP, INF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL_MASTER = float(os.environ.get("TL_MASTER", 2700)); GAP_MASTER = float(os.environ.get("GAP_MASTER", 2e-4))


def worst_p(Q, p0, th1, thi):
    """式(37)：max Σ p_s Q_s，s.t. Σp = 1，Σ|p − p0| ≤ θ1，|p_s − p0_s| ≤ θ∞，p ≥ 0。变量 [p, p+, p−]。"""
    if th1 <= 0 and thi <= 0: return np.asarray(p0, float).copy()
    K = len(Q); I = np.eye(K); Z = np.zeros((K, K))
    c = np.r_[-np.asarray(Q, float), np.zeros(2 * K)]
    Aeq = np.vstack([np.r_[np.ones(K), np.zeros(2 * K)], np.hstack([I, -I, I])]); beq = np.r_[1.0, p0]
    Aub = np.vstack([np.r_[np.zeros(K), np.ones(2 * K)], np.hstack([Z, I, I])]); bub = np.r_[th1, np.full(K, thi)]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (3 * K), method="highs")
    p = np.clip(r.x[:K], 0, None); return p / p.sum()


def build_master(P, D, flags, days, int_add=None):
    M = MILP(); X, a = BO.add_capacity(M, P, flags, integer_N=False)
    Td = BO.tamb_design(D, P); Vs = []
    for s, i in enumerate(days):
        Vs.append(BO.add_day(M, P, X, BO.day_of(D, i), flags, w=0.0, Tamb_des=Td, int_add=(int_add or {}).get(s)))
    return M, X, Vs, a


def solve_sub(P, D, flags, x, days, int_add=None, tl=600, gap=1e-4):
    Td = BO.tamb_design(D, P); Q = []; R = []
    for s, i in enumerate(days):
        M = MILP(); X, _ = BO.add_capacity(M, P, flags, fix=x)
        V = BO.add_day(M, P, X, BO.day_of(D, i), flags, w=1.0, Tamb_des=Td, int_add=(int_add or {}).get(s))
        sol, st, obj, g = M.solve(gap=gap, tl=tl)
        if sol is None: raise RuntimeError(f"子问题不可行: 日 {i}, 状态 {st}")
        Q.append(obj); R.append(BO.extract(sol, V))
    return np.array(Q), R


def simul_check(R, tol=1.0):
    """同时充放电(b) / 同时充放冷(z) / 同时购售电(g) 的时段，返回 {场景序号: {变量: [时段]}}。"""
    out = {}
    for s, r in enumerate(R):
        d = {}
        for var, a, b in (("b", "Pch", "Pdis"), ("z", "qch", "qdis"), ("g", "buy", "sell")):
            ts = np.where(np.minimum(r[a], r[b]) > tol)[0]
            if len(ts): d[var] = ts.tolist()
        if d: out[s] = d
    return out


def start_from(M, X, Vs, r):
    """由已有结果（容量 x 与各典型日运行解 R）构造完整初始解向量，用于热启动（要求模型结构一致、可行域包含该解）。"""
    v = np.zeros(M.n)
    for c in BO.CAPS: v[X[c]] = r["x"][c]
    for V, R in zip(Vs, r["R"]):
        for k, idx in V.items():
            if k.startswith("_"): continue
            v[idx] = R[k]
    return v


def load_init():
    f = os.environ.get("INIT_FROM")
    return pickle.load(open(f, "rb")) if f else None


def finalize(P, D, flags, x, days, p0, th1, thi, int_add, a):
    """N 取整后固定全部容量，重算运行成本与最恶劣分布。"""
    x = dict(x); x["N"] = float(round(x["N"]))
    Q, R = solve_sub(P, D, flags, x, days, int_add)
    p = worst_p(Q, p0, th1, thi); inv = sum(a[c] * x[c] for c in BO.CAPS)
    return dict(x=x, Q=Q, R=R, p=p, inv=inv, obj=inv + P["DAYS"] * float(p @ Q))


def ccg(P, D, scen, th, eps=1e-3, max_it=10, log=print, int_add=None):
    flags = BO.flags_of(scen); days = list(D["med"]); p0 = D["p0"]; th1, thi = th
    M, X, Vs, a = build_master(P, D, flags, days, int_add)
    eta = M.var(1, -INF, INF, cost=1.0)
    cuts = []; LB, UB = -np.inf, np.inf; hist = []; best = None; t0 = time.time(); start = None
    init = load_init()
    if init is not None: start = start_from(M, X, Vs, init)
    p = p0.copy()
    for k in range(1, max_it + 1):
        cuts.append(p)
        M.row([(eta, 1.0)] + [(V["Q"], -P["DAYS"] * p[s]) for s, V in enumerate(Vs)], lo=0.0)
        if start is not None:   # 热启动：上一轮解，η 取满足全部割的最小值
            qv = np.array([start[V["Q"]][0] for V in Vs])
            start[eta] = max(P["DAYS"] * float(c @ qv) for c in cuts)
        sol, st, obj, gabs = M.solve(gap=GAP_MASTER, tl=TL_MASTER, start=start)
        if sol is None: raise RuntimeError(f"主问题失败: {st}")
        start = sol.copy()
        LB = max(LB, M.dual_bound)
        x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
        inv = sum(a[c] * x[c] for c in BO.CAPS)
        Q, R = solve_sub(P, D, flags, x, days, int_add)
        p = worst_p(Q, p0, th1, thi)
        ub_k = inv + P["DAYS"] * float(p @ Q)
        if ub_k < UB: UB = ub_k; best = dict(x=x)
        gap = (UB - LB) / abs(UB)
        hist.append(dict(it=k, LB=LB, UB=UB, gap=gap, t=time.time() - t0, master_status=st, master_gap=gabs / abs(obj)))
        log(f"[{scen} θ=({th1:.3g},{thi:.3g})] 迭代 {k}: LB {LB / 1e4:.1f} 万元, UB {UB / 1e4:.1f} 万元, gap {gap:.2e}, "
            f"主问题gap {gabs / abs(obj):.1e}, N {x['N']:.1f}, FC {x['FC']:.0f}, MC {x['MC']:.0f}, TES {x['TES']:.0f}, HP {x['HP']:.0f}, "
            f"PV {x['PV']:.0f}, WT {x['WT']:.0f}, EB {x['EB']:.0f}, PB {x['PB']:.0f}, 用时 {time.time() - t0:.0f} s", flush=True)
        if gap <= eps: break
        if any(np.allclose(p, c, atol=1e-6) for c in cuts): break
    fin = finalize(P, D, flags, best["x"], days, p0, th1, thi, int_add, a)
    fin.update(hist=hist, scen=scen, theta=(th1, thi), days=days, p0=p0, a=a, method="ccg", x_cont=best["x"],
               t=time.time() - t0, LB=LB, gap_final=(fin["obj"] - LB) / abs(fin["obj"]))
    log(f"[{scen}] 取整 N={fin['x']['N']:.0f} 后最终目标 {fin['obj'] / 1e4:.1f} 万元，相对 LB gap {fin['gap_final']:.2e}", flush=True)
    return fin


def mono(P, D, scen, th, log=print, int_add=None, tl=None, gap=None):
    """等价单层 MILP（(37)的对偶）。"""
    flags = BO.flags_of(scen); days = list(D["med"]); p0 = D["p0"]; th1, thi = th; K = len(days); t0 = time.time()
    M, X, Vs, a = build_master(P, D, flags, days, int_add)
    lam = M.var(1, -INF, INF, cost=1.0); pi = M.var(K, -INF, INF, cost=p0)
    mu = M.var(1, 0.0, INF, cost=th1); nu = M.var(K, 0.0, INF, cost=thi)
    for s, V in enumerate(Vs):
        M.row([(lam, 1.0), (pi[s], 1.0), (V["Q"], -P["DAYS"])], lo=0.0)
        M.row([(pi[s], 1.0), (mu, -1.0), (nu[s], -1.0)], hi=0.0)
        M.row([(pi[s], 1.0), (mu, 1.0), (nu[s], 1.0)], lo=0.0)
    init = load_init(); start = None
    if init is not None:     # 可行初值：μ = ν = π = 0，λ = 365·max_s Q_s
        start = start_from(M, X, Vs, init)
        start[lam] = P["DAYS"] * max(start[V["Q"]][0] for V in Vs)
    sol, st, obj, gabs = M.solve(gap=gap or GAP_MASTER, tl=tl or TL_MASTER, start=start)
    x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
    fin = finalize(P, D, flags, x, days, p0, th1, thi, int_add, a)
    fin.update(hist=[dict(it=1, LB=M.dual_bound, UB=obj, gap=gabs / abs(obj), t=time.time() - t0, master_status=st)], scen=scen,
               theta=(th1, thi), days=days, p0=p0, a=a, method="mono", x_cont=x, t=time.time() - t0, LB=M.dual_bound,
               obj_milp=obj, gap_final=(fin["obj"] - M.dual_bound) / abs(fin["obj"]))
    log(f"[{scen} 单层MILP θ=({th1:.3g},{thi:.3g})] 状态 {st}, 目标 {obj / 1e4:.1f} 万元, best bound {M.dual_bound / 1e4:.1f}, "
        f"gap {gabs / abs(obj):.2e}; 取整后 {fin['obj'] / 1e4:.1f} 万元, N {fin['x']['N']:.0f}, FC {x['FC']:.0f}, MC {x['MC']:.0f}, "
        f"TES {x['TES']:.0f}, HP {x['HP']:.0f}, PV {x['PV']:.0f}, WT {x['WT']:.0f}, EB {x['EB']:.0f}, PB {x['PB']:.0f}, 用时 {time.time() - t0:.0f} s", flush=True)
    return fin


def run(site, scen, tag, method="ccg", over=None):
    P = PR.params(**(over or {})); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    th = {"SP": (0.0, 0.0), "RO": (2.0, 1.0)}.get(tag.split("_")[0]) or PR.theta(P)
    f = mono if method == "mono" else ccg
    int_add = None
    for rnd in range(3):            # 校验同时充放/购售，出现则只在相应场景、时段加回 0-1 变量后重解
        r = f(P, D, scen, th, int_add=int_add)
        sc = simul_check(r["R"])
        r["simul"] = sc; r["int_add"] = int_add
        if not sc: break
        print(f"[{scen}] 发现同时充放/购售: {sc}，加回对应 0-1 变量重解", flush=True)
        int_add = int_add or {}
        for s, d in sc.items():
            for var, ts in d.items():
                int_add.setdefault(s, {}).setdefault(var, set()).update(ts)
    r["params_over"] = over or {}
    pickle.dump(r, open(f"{ROOT}/results/{site}_{tag}{'_mono' if method == 'mono' else ''}.pkl", "wb"))
    return r


if __name__ == "__main__":
    site, scen, tag = sys.argv[1:4]
    method = sys.argv[4] if len(sys.argv) > 4 else "ccg"
    over = json.loads(os.environ["PARAM_OVER"]) if os.environ.get("PARAM_OVER") else None
    run(site, scen, tag, method, over)
