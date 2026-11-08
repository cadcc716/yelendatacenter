"""
两阶段分布鲁棒容量规划 式(32)–(40)：两步法 + 嵌套热启动
  第一步：以总成本为目标（N 取连续变量），求得 GPU 数量 N*（取整）
  第二步：固定 N = N*，去掉 GPU 年化投资常数项，只以能源侧年化成本为目标再求解；
          MIP 相对间隙与 C&CG 收敛判据 ε 均相对能源侧成本计算
  嵌套热启动：S4 ← S3（E_TES = 0），S5 ← S4（K_HP = 0），S2 ← S1；上一场景的能源侧成本作为截断值，
          并作为本场景上界的初值，保证 S5 ≤ S4 ≤ S3、S2 ≤ S1
  C&CG：主问题(35)（上一轮解热启动，LB 取 best bound）、子问题(36)(37)；最恶劣分布与已有割重复即判收敛
  等价单层 MILP(40)：(37) 的对偶并入外层最小化
  θ1 = θ∞ = 0 → SP；θ1 = 2、θ∞ = 1 → RO
用法：python ccg_solve.py <site> <开关场景S1–S5> <结果标签> [ccg|mono] [上一场景结果pkl]
环境变量：GAP1/TL1（第一步）、GAP2/TL2（第二步）、EPS、STEP1（ccg|mono，第一步解法）、PARAM_OVER（JSON 参数覆盖）
"""
import os, sys, time, pickle, json
import numpy as np
from scipy.optimize import linprog
import params as PR, build_op as BO
from milp import MILP, INF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = lambda k, d: type(d)(os.environ.get(k, d))


def worst_p(Q, p0, th1, thi):
    """式(37)：max Σ p_s Q_s，s.t. Σp = 1，Σ|p − p0| ≤ θ1，|p_s − p0_s| ≤ θ∞，p ≥ 0。变量 [p, p+, p−]。"""
    if th1 <= 0 and thi <= 0: return np.asarray(p0, float).copy()
    K = len(Q); I = np.eye(K); Z = np.zeros((K, K))
    c = np.r_[-np.asarray(Q, float), np.zeros(2 * K)]
    Aeq = np.vstack([np.r_[np.ones(K), np.zeros(2 * K)], np.hstack([I, -I, I])]); beq = np.r_[1.0, p0]
    Aub = np.vstack([np.r_[np.zeros(K), np.ones(2 * K)], np.hstack([Z, I, I])]); bub = np.r_[th1, np.full(K, thi)]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (3 * K), method="highs")
    p = np.clip(r.x[:K], 0, None); return p / p.sum()


def build_master(P, D, flags, days, int_add=None, fixN=None):
    M = MILP(); X, a = BO.add_capacity(M, P, flags, integer_N=False, fixN=fixN)
    if fixN is not None: a = dict(a, N=0.0)          # 第二步：GPU 投资为常数，不计入目标
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
    """同时充放电(b) / 同时充放冷(z) 的时段，返回 {典型日序号: {变量: [时段]}}。"""
    out = {}
    for s, r in enumerate(R):
        d = {}
        for var, a, b in (("b", "Pch", "Pdis"), ("z", "qch", "qdis")):
            ts = np.where(np.minimum(r[a], r[b]) > tol)[0]
            if len(ts): d[var] = ts.tolist()
        if d: out[s] = d
    return out


def start_from(M, X, Vs, r):
    """由已有结果（容量 x 与各典型日运行解 R）构造完整初始解向量（要求本模型可行域包含该解）。"""
    v = np.zeros(M.n)
    for c in BO.CAPS: v[X[c]] = r["x"].get(c, 0.0)
    for V, R in zip(Vs, r["R"]):
        for k, idx in V.items():
            if k.startswith("_") or k not in R: continue
            v[idx] = R[k]
    return v


def evaluate(P, D, flags, x, days, p0, th, int_add, a):
    """固定全部容量，逐典型日求运行成本，得最恶劣分布与目标值。"""
    Q, R = solve_sub(P, D, flags, x, days, int_add)
    p = worst_p(Q, p0, *th); inv = sum(a[c] * x[c] for c in BO.CAPS)
    return dict(x=dict(x), Q=Q, R=R, p=p, inv=inv, obj=inv + P["DAYS"] * float(p @ Q))


def ccg(P, D, scen, th, fixN=None, int_add=None, init=None, cutoff=None, seed=None, gap=2e-4, tl=2700, eps=1e-3, max_it=10, log=print):
    flags = BO.flags_of(scen); days = list(D["med"]); p0 = D["p0"]; th1, thi = th
    M, X, Vs, a = build_master(P, D, flags, days, int_add, fixN)
    eta = M.var(1, -INF, INF, cost=1.0)
    cuts = []; LB, UB = -np.inf, np.inf; hist = []; best = None; t0 = time.time()
    start = start_from(M, X, Vs, init) if init is not None else None
    if seed is not None: UB, best = seed["obj"], seed          # 上一场景解在本场景可行，作为上界初值
    p = p0.copy()
    for k in range(1, max_it + 1):
        cuts.append(p)
        M.row([(eta, 1.0)] + [(V["Q"], -P["DAYS"] * p[s]) for s, V in enumerate(Vs)], lo=0.0)
        if start is not None:   # η 取满足全部割的最小值
            qv = np.array([start[V["Q"]][0] for V in Vs])
            start[eta] = max(P["DAYS"] * float(c @ qv) for c in cuts)
        sol, st, obj, gabs = M.solve(gap=gap, tl=tl, start=start, cutoff=cutoff)
        if sol is None:
            log(f"[{scen}] 主问题未得到优于截断值的解（{st}），沿用上一场景解", flush=True)
            LB = max(LB, M.dual_bound if np.isfinite(M.dual_bound) else LB); break
        start = sol.copy(); LB = max(LB, M.dual_bound)
        x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
        if fixN is None: x["N"] = x["N"]                     # 第一步 N 连续
        ev = evaluate(P, D, flags, x, days, p0, th, int_add, a)
        if ev["obj"] < UB: UB, best = ev["obj"], ev
        p = worst_p(ev["Q"], p0, th1, thi)
        gapk = (UB - LB) / abs(UB)
        hist.append(dict(it=k, LB=LB, UB=UB, gap=gapk, t=time.time() - t0, master_status=st, master_gap=gabs / max(abs(obj), 1e-9)))
        log(f"[{scen} θ=({th1:.3g},{thi:.3g}) N{'*=' + str(int(fixN)) if fixN is not None else '连续'}] 迭代 {k}: LB {LB / 1e4:.2f} 万元, "
            f"UB {UB / 1e4:.2f} 万元, gap {gapk:.2e}, 主问题gap {gabs / max(abs(obj), 1e-9):.1e}, N {x['N']:.1f}, FC {x['FC']:.0f}, "
            f"MC {x['MC']:.0f}, TES {x['TES']:.0f}, HP {x['HP']:.0f}, PV {x['PV']:.0f}, EB {x['EB']:.0f}, PB {x['PB']:.0f}, "
            f"用时 {time.time() - t0:.0f} s", flush=True)
        if gapk <= eps: break
        if any(np.allclose(p, c, atol=1e-6) for c in cuts): break     # 最恶劣分布重复，判定收敛
    best = dict(best); best.update(hist=hist, LB=LB, gap_final=(best["obj"] - LB) / abs(best["obj"]) if np.isfinite(LB) else np.nan,
                                   t=time.time() - t0, a=a, method="ccg")
    return best


def mono(P, D, scen, th, fixN=None, int_add=None, init=None, cutoff=None, seed=None, gap=2e-4, tl=2700, log=print, **_):
    flags = BO.flags_of(scen); days = list(D["med"]); p0 = D["p0"]; th1, thi = th; K = len(days); t0 = time.time()
    M, X, Vs, a = build_master(P, D, flags, days, int_add, fixN)
    lam = M.var(1, -INF, INF, cost=1.0); pi = M.var(K, -INF, INF, cost=p0)
    mu = M.var(1, 0.0, INF, cost=th1); nu = M.var(K, 0.0, INF, cost=thi)
    for s, V in enumerate(Vs):
        M.row([(lam, 1.0), (pi[s], 1.0), (V["Q"], -P["DAYS"])], lo=0.0)
        M.row([(pi[s], 1.0), (mu, -1.0), (nu[s], -1.0)], hi=0.0)
        M.row([(pi[s], 1.0), (mu, 1.0), (nu[s], 1.0)], lo=0.0)
    start = None
    if init is not None:     # 可行初值：μ = ν = π = 0，λ = 365·max_s Q_s
        start = start_from(M, X, Vs, init); start[lam] = P["DAYS"] * max(start[V["Q"]][0] for V in Vs)
    sol, st, obj, gabs = M.solve(gap=gap, tl=tl, start=start, cutoff=cutoff)
    if sol is None:
        log(f"[{scen} 单层] 未得到优于截断值的解（{st}），沿用上一场景解", flush=True)
        best = dict(seed); LB = M.dual_bound
    else:
        x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
        best = evaluate(P, D, flags, x, days, p0, th, int_add, a); LB = M.dual_bound
        if seed is not None and seed["obj"] < best["obj"]: best = dict(seed)
    best.update(hist=[dict(it=1, LB=LB, UB=obj, gap=gabs / max(abs(obj), 1e-9), t=time.time() - t0, master_status=st)], LB=LB,
                gap_final=(best["obj"] - LB) / abs(best["obj"]), t=time.time() - t0, a=a, method="mono", obj_milp=obj)
    log(f"[{scen} 单层MILP θ=({th1:.3g},{thi:.3g}) N{'*=' + str(int(fixN)) if fixN is not None else '连续'}] 状态 {st}, "
        f"MILP目标 {obj / 1e4:.2f} 万元, best bound {LB / 1e4:.2f}, 最终 {best['obj'] / 1e4:.2f} 万元, gap {best['gap_final']:.2e}, "
        f"N {best['x']['N']:.1f}, FC {best['x']['FC']:.0f}, MC {best['x']['MC']:.0f}, TES {best['x']['TES']:.0f}, HP {best['x']['HP']:.0f}, "
        f"PV {best['x']['PV']:.0f}, EB {best['x']['EB']:.0f}, PB {best['x']['PB']:.0f}, 用时 {time.time() - t0:.0f} s", flush=True)
    return best


def merge_int(a, b):
    """合并两组需加回 0-1 变量的 {典型日: {变量: [时段]}}。"""
    out = {k: {v: list(h) for v, h in d.items()} for k, d in (a or {}).items()}
    for k, d in (b or {}).items():
        for v, h in d.items(): out.setdefault(k, {}).setdefault(v, []); out[k][v] = sorted(set(out[k][v]) | set(h))
    return out


def fix_simul(P, D, flags, ev, days, p0, th, int_add, a, log=print):
    """邻域搜索等固定容量评估后若出现同时充放，加回相应 0-1 变量重新逐日求解，直至消除。"""
    sc = simul_check(ev["R"])
    for _ in range(4):
        if not sc: break
        int_add = merge_int(int_add, sc); log(f"  校验发现同时充放 {list(sc)}，加回 0-1 变量重算", flush=True)
        ev = evaluate(P, D, flags, ev["x"], days, p0, th, int_add, a); sc = simul_check(ev["R"])
    ev["simul"] = sc; ev["int_add"] = int_add
    return ev


def with_check(f, P, D, scen, th, log=print, **kw):
    """求解后校验同时充放电/充放冷，出现则只在相应典型日、时段加回 0-1 变量重解。"""
    int_add = None
    for _ in range(3):
        r = f(P, D, scen, th, int_add=int_add, log=log, **kw)
        sc = simul_check(r["R"]); r["simul"] = sc; r["int_add"] = int_add
        if not sc: return r
        log(f"[{scen}] 发现同时充放: {sc}，加回对应 0-1 变量重解", flush=True)
        int_add = int_add or {}
        for s, d in sc.items():
            for var, ts in d.items(): int_add.setdefault(s, {}).setdefault(var, set()).update(ts)
    return r


def two_step(site, scen, tag, method="ccg", prev=None, over=None, log=print):
    P = PR.params(**(over or {})); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    if P.get("R_N", 0.0) > 0:                        # 算力保障约束：N ≥ (1+R_N)·全年日均满频算力需求
        P["N_MIN"] = float(np.ceil((1 + P["R_N"]) * (D["AR"] + D["AB"]).mean()))
        log(f"[{scen}] 算力保障约束 N ≥ {P['N_MIN']:.0f}（冗余率 {P['R_N']:.0%}）", flush=True)
    th = {"SP": (0.0, 0.0), "RO": (2.0, 1.0)}.get(tag.split("_")[0]) or PR.theta(P)
    f1 = mono if ENV("STEP1", method) == "mono" else ccg
    f2 = mono if method == "mono" else ccg
    # 第一步：总成本，求 N*
    if os.environ.get("FIXN") and ENV("SKIP1", 0) == 1:       # 筛选试算：跳过第一步
        r1 = dict(x={k: 0.0 for k in BO.CAPS}, obj=float("nan"), gap_final=float("nan"), t=0.0, hist=[])
    else:
        r1 = with_check(f1, P, D, scen, th, log=log, gap=ENV("GAP1", 3e-3), tl=ENV("TL1", 1500.0))
    Nstar = float(round(r1["x"]["N"]))
    if os.environ.get("FIXN"):                      # GPU 冗余度试验：第二步把 GPU 数量固定为给定值
        Nstar = float(os.environ["FIXN"]); log(f"[{scen}] GPU 数量按设定固定为 {Nstar:.0f}", flush=True)
    log(f"[{scen}] 第一步：N* = {Nstar:.0f}，总成本 {r1['obj'] / 1e4:.2f} 万元", flush=True)
    # 第二步：固定 N*，能源侧成本；嵌套热启动与截断
    init = cutoff = seed = None
    if prev is not None:
        pr = pickle.load(open(prev, "rb"))
        if abs(pr["x"]["N"] - Nstar) < 0.5:
            flags = BO.flags_of(scen); a2 = dict(PR.annual_cost(P), N=0.0)
            init = pr; cutoff = pr["obj"] * (1 + 1e-7) + 1.0
            seed = dict(x=dict(pr["x"]), Q=pr["Q"], R=pr["R"], p=pr["p"], inv=pr["inv"], obj=pr["obj"])
            log(f"[{scen}] 嵌套热启动：初始解与截断值取自 {os.path.basename(prev)}（能源侧 {pr['obj'] / 1e4:.2f} 万元）", flush=True)
        else:
            log(f"[{scen}] 上一场景 N*={pr['x']['N']:.0f} 与本场景 N*={Nstar:.0f} 不同，不做嵌套热启动", flush=True)
    # 第一步容量（N 取 N*）作为第二步的候选初值，与上一场景解比较取优
    a2 = dict(PR.annual_cost(P), N=0.0); x1 = dict(r1["x"], N=Nstar)
    if not np.isnan(r1["obj"]):
        c1 = evaluate(P, D, BO.flags_of(scen), x1, list(D["med"]), D["p0"], th, None, a2)
        log(f"[{scen}] 第一步容量在第二步口径下的能源侧成本 {c1['obj'] / 1e4:.2f} 万元", flush=True)
        if seed is None or c1["obj"] < seed["obj"]:
            seed = dict(c1); init = c1; cutoff = c1["obj"] * (1 + 1e-7) + 1.0
    r2 = with_check(f2, P, D, scen, th, log=log, fixN=Nstar, init=init, cutoff=cutoff, seed=seed,
                    gap=ENV("GAP2", 1e-3), tl=ENV("TL2", 2700.0), eps=ENV("EPS", 1e-3))
    if seed is not None and r2["obj"] > seed["obj"] * (1 + 1e-6):          # 单调性校验
        log(f"[{scen}] 能源侧成本 {r2['obj'] / 1e4:.2f} 高于上一场景 {seed['obj'] / 1e4:.2f}，延长时限重解", flush=True)
        r2b = with_check(f2, P, D, scen, th, log=log, fixN=Nstar, init=init, cutoff=cutoff, seed=seed,
                         gap=ENV("GAP2", 1e-3), tl=2 * ENV("TL2", 2700.0), eps=ENV("EPS", 1e-3))
        r2 = r2b if r2b["obj"] <= r2["obj"] else r2
    r2["obj_pre_ls"] = r2["obj"]; r2["x_pre_ls"] = dict(r2["x"])
    if ENV("LS", 1) == 1:                       # 容量邻域搜索改进初始可行解（最优性仍以 MILP 下界衡量）
        import local_search as LS
        imp, hist = LS.search(P, D, scen, th, r2, list(D["med"]), D["p0"], r2["a"], log=log, worst_p=worst_p,
                              max_eval=ENV("LS_MAX", 60))
        r2["ls_hist"] = hist
        if imp is not None:
            ev = evaluate(P, D, BO.flags_of(scen), imp["x"], list(D["med"]), D["p0"], th, r2.get("int_add"), r2["a"])
            ev = fix_simul(P, D, BO.flags_of(scen), ev, list(D["med"]), D["p0"], th, r2.get("int_add"), r2["a"], log=log)
            log(f"[{scen}] 邻域搜索改进：{r2['obj'] / 1e4:.2f} → {ev['obj'] / 1e4:.2f} 万元", flush=True)
            r2.update(ev); r2["gap_final"] = (r2["obj"] - r2["LB"]) / abs(r2["obj"])
    r2.update(step1=dict(N=Nstar, obj_total=r1["obj"], x=r1["x"], gap=r1["gap_final"], t=r1["t"], hist=r1["hist"]),
              scen=scen, theta=th, days=list(D["med"]), p0=D["p0"], params_over=over or {}, prev=prev,
              a_full=PR.annual_cost(P))
    log(f"[{scen}] 第二步：能源侧年化成本 {r2['obj'] / 1e4:.2f} 万元，gap {r2['gap_final']:.2e}", flush=True)
    return r2


if __name__ == "__main__":
    site, scen, tag = sys.argv[1:4]
    method = sys.argv[4] if len(sys.argv) > 4 else "ccg"
    prev = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    over = json.loads(os.environ["PARAM_OVER"]) if os.environ.get("PARAM_OVER") else None
    r = two_step(site, scen, tag, method, prev, over)
    out = os.environ.get("OUT") or f"{ROOT}/results/{site}_{tag}{'_mono' if method == 'mono' else ''}.pkl"
    pickle.dump(r, open(out, "wb"))
