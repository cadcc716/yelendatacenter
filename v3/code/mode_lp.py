"""固定温区模式的规划 LP 与单日 LP（执行方案 2.3、2.4(b)(c)）。

ctx（上下文）统一从已有结果 pkl 读取，保证与原结果口径一致：
    ctx = make_ctx("zhongwei_DRO_S5")             # DRO
    ctx = make_ctx("zhongwei_DRO_S5", th=(0, 0))  # SP；RO 取 th=(2, 1)
"""
import os, pickle, numpy as np
from multiprocessing import Pool
import params as PR, build_op as BO, ccg_solve as C
from milp import MILP, INF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECAPS = [c for c in BO.CAPS if c != "N"]          # 能源侧设备（N 由算力保障约束固定）


def make_ctx(src="zhongwei_DRO_S5", th=None, site=None, rdir=None, scen="S5"):
    rdir = rdir or f"{ROOT}/results"
    r = pickle.load(open(f"{rdir}/{src}.pkl", "rb"))
    site = site or src.split("_DRO")[0].split("_SP")[0].split("_RO")[0]
    P = PR.params(**(r.get("params_over") or {})); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    return dict(P=P, D=D, flags=BO.flags_of(scen), days=list(r["days"]), p0=np.asarray(r["p0"], float),
                th=tuple(r["theta"]) if th is None else tuple(th), N=float(r["x"]["N"]),
                a=dict(PR.annual_cost(P), N=0.0), Td=BO.tamb_design(D, P), src=src, r=r)


def dro_rows(M, ctx, Qs):
    """与 ccg_solve.mono 相同的 DRO 对偶行：max_p Σ p_s·365·Q_s 的对偶。返回对偶变量索引。"""
    th1, thi = ctx["th"]; K = len(Qs); P = ctx["P"]
    lam = M.var(1, -INF, INF, cost=1.0); pi = M.var(K, -INF, INF, cost=ctx["p0"])
    mu = M.var(1, 0.0, INF, cost=th1); nu = M.var(K, 0.0, INF, cost=thi)
    for s, q in enumerate(Qs):
        M.row([(lam, 1.0), (pi[s], 1.0), (q, -P["DAYS"])], lo=0.0)
        M.row([(pi[s], 1.0), (mu, -1.0), (nu[s], -1.0)], hi=0.0)
        M.row([(pi[s], 1.0), (mu, 1.0), (nu[s], 1.0)], lo=0.0)
    return dict(lam=lam, pi=pi, mu=mu, nu=nu)


def build(ctx, spec, copies=True, int_add=None):
    """规划模型（N 固定、能源侧口径）。spec[s] 为第 s 日的温区设定：
         np.ndarray (J,T) → 温区固定（u_mode）；"free" → u 连续（由调用方加模式一致性约束）；"int" → u 为 0-1。
       copies=True 时为每天建容量副本 X_s 并记录连接行 X_s − x = 0 的行号（取对偶用）。"""
    P, fl = ctx["P"], ctx["flags"]; xm = PR.xmax(P)
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, integer_N=False, fixN=ctx["N"])
    Vs, Xs, link = [], [], []
    for s, i in enumerate(ctx["days"]):
        Xd = X
        if copies:
            Xd = {"N": X["N"]}; rows = {}
            for c in ECAPS:
                ub = xm[c] if not ((c == "TES" and not fl["tes"]) or (c == "HP" and not fl["heat"])) else 0.0
                Xd[c] = M.var(1, 0.0, ub); rows[c] = M.nr
                M.row([(Xd[c], 1.0), (X[c], -1.0)], lo=0.0, hi=0.0)
            link.append(rows)
        sp = spec[s]; kw = {}
        if isinstance(sp, str): kw = dict(u_int=(sp == "int"))
        else: kw = dict(u_mode=sp)
        Vs.append(BO.add_day(M, P, Xd, BO.day_of(ctx["D"], i), fl, w=0.0, Tamb_des=ctx["Td"],
                             int_add=(int_add or {}).get(s), **kw))
        Xs.append(Xd)
    dv = dro_rows(M, ctx, [V["Q"] for V in Vs])
    return M, X, Xs, Vs, link, dv


def _solve_fixed(ctx, modes_by_day, int_add=None, tl=600):
    M, X, Xs, Vs, link, dv = build(ctx, modes_by_day, copies=True, int_add=int_add)
    sol, st, obj, g = M.solve(gap=1e-6, tl=tl)
    if sol is None: return None
    x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
    Q = np.array([float(sol[V["Q"]][0]) for V in Vs]); R = [BO.extract(sol, V) for V in Vs]
    pi = None
    if M.row_dual is not None:
        pi = [{c: float(M.row_dual[rows[c]]) for c in ECAPS} for rows in link]
    return dict(x=x, Q=Q, R=R, obj=obj, pi=pi, sol=sol, status=st)


def dro_values(ctx, Q):
    """给定各日成本 Q，求 DRO 对偶变量 (λ, π, μ, ν) 的取值（构造热启动解用）。"""
    M = MILP(); q = [M.var(1, float(v), float(v)) for v in Q]; dv = dro_rows(M, ctx, q)
    sol, st, obj, g = M.solve(gap=1e-9, tl=60)
    return {k: sol[idx] for k, idx in dv.items()}, obj


def eval_fixed(ctx, x, modes_by_day, int_add=None, log=None):
    """固定容量 x 与各日模式，逐日求运行成本（含同时充放校验：出现则该日整日加回 0-1 变量重解）。
       返回的目标是 K-模式模型在该 (x, 模式) 下的精确值。"""
    T = ctx["P"]["T"]; int_add = {k: dict(v) for k, v in (int_add or {}).items()}
    Q, R = [], []
    for s, U in enumerate(modes_by_day):
        for _ in range(4):
            q, r = day_cost(ctx, x, s, U, int_add=int_add.get(s), ret_R=True)
            sc = C.simul_check([r]) if r is not None else {}
            if not sc: break
            int_add = C.merge_int(int_add, {s: {v: list(range(T)) for v in sc[0]}})
            if log: log(f"    日 {s} 同时充放 {list(sc[0])}，整日加回 0-1 变量", flush=True)
        Q.append(q); R.append(r)
    Q = np.array(Q); p = C.worst_p(Q, ctx["p0"], *ctx["th"])
    return dict(x=dict(x), Q=Q, R=R, p=p, obj=inv_of(ctx, x) + ctx["P"]["DAYS"] * float(p @ Q),
                int_add=int_add, simul=C.simul_check(R))


def _start_vec(ctx, M, X, Xs, Vs, dv, ev):
    v = np.zeros(M.n)
    for c in BO.CAPS: v[X[c]] = ev["x"][c]
    for Xd in Xs:
        for c in ECAPS: v[Xd[c]] = ev["x"][c]
    for V, R in zip(Vs, ev["R"]):
        for k, idx in V.items():
            if not k.startswith("_") and k in R: v[idx] = R[k]
    vals, _ = dro_values(ctx, ev["Q"])
    for k, idx in dv.items(): v[idx] = vals[k]
    return v


def lp_fixed(ctx, modes_by_day, int_add=None, check=True, reopt_tl=300, log=None):
    """各日温区模式固定 → 关于容量与运行的 LP。返回 x、各日 Q、目标值、最恶劣分布 p、连接行对偶 pi[s][c]。
       check=True 时做同时充放校验：出现则 (1) 固定容量逐日加回 0-1 变量重解（精确上界）；
       (2) 以此为热启动、在加回 0-1 变量的规划 MILP 上限时重定容量，取较优者。对偶沿用 LP。"""
    res = _solve_fixed(ctx, modes_by_day, int_add)
    if res is None: return None
    pi = res["pi"]; lp_obj = res["obj"]
    sc = C.simul_check(res["R"]) if check else {}
    out = dict(res, p=C.worst_p(res["Q"], ctx["p0"], *ctx["th"]), simul=sc, int_add=int_add, obj_lp=lp_obj, how="LP")
    if not sc: return out
    if log: log(f"    LP 解同时充放 {sc}，校验修正", flush=True)
    ev = eval_fixed(ctx, res["x"], modes_by_day, int_add, log)
    best = dict(ev, how="固定容量逐日加回0-1")
    if reopt_tl and reopt_tl > 0:
        M, X, Xs, Vs, link, dv = build(ctx, modes_by_day, copies=True, int_add=ev["int_add"])
        sol, st, obj, g = M.solve(gap=1e-4, tl=reopt_tl, start=_start_vec(ctx, M, X, Xs, Vs, dv, ev))
        if sol is not None and obj < best["obj"] - 1.0:
            x = {c: float(sol[X[c]][0]) for c in BO.CAPS}
            ev2 = eval_fixed(ctx, x, modes_by_day, ev["int_add"], log)     # 重定容量后再逐日校验，取精确值
            if ev2["obj"] < best["obj"]: best = dict(ev2, how=f"加回0-1后重定容量（{st}）")
    return dict(best, pi=pi, obj_lp=lp_obj)


def inv_of(ctx, x):
    return sum(ctx["a"][c] * x[c] for c in BO.CAPS)


def day_cost(ctx, x, s, U, int_add=None, ret_R=False):
    """固定容量 x 与温区模式 U 的单日 LP（s 为典型日序号或日数据字典）；不可行返回 inf。"""
    P = ctx["P"]; M = MILP(); X, _ = BO.add_capacity(M, P, ctx["flags"], fix=x)
    day = BO.day_of(ctx["D"], ctx["days"][s]) if np.isscalar(s) else s
    V = BO.add_day(M, P, X, day, ctx["flags"], w=1.0, Tamb_des=ctx["Td"], int_add=int_add, u_mode=U)
    sol, st, obj, g = M.solve(gap=1e-6, tl=120)
    if sol is None: return (np.inf, None) if ret_R else np.inf
    return (obj, BO.extract(sol, V)) if ret_R else obj


_G = {}
def _dc(args):
    s, w = args
    return day_cost(_G["ctx"], _G["x"], s, _G["pool"][w])


def cost_matrix(ctx, x, pool, procs=3, mask=None):
    """C[s, w] = day_cost(x, s, pool[w])。mask[s, w]=False 的组合不算，记为 inf。"""
    S, W = len(ctx["days"]), len(pool)
    jobs = [(s, w) for s in range(S) for w in range(W) if mask is None or mask[s, w]]
    _G.update(ctx=ctx, x=x, pool=pool)
    if procs > 1:
        with Pool(procs) as pp: vals = pp.map(_dc, jobs, chunksize=4)
    else: vals = list(map(_dc, jobs))
    Cm = np.full((S, W), np.inf)
    for (s, w), v in zip(jobs, vals): Cm[s, w] = v
    return Cm
