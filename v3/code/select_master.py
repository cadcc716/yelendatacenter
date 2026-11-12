"""模式池上的模式选址（执行方案 1.5 节、2.4(d)(d′)）。

select_fast：成本矩阵选模式 ↔ LP 重定容量的交替（块坐标下降，默认使用）；
select     ：模式选址主问题 MILP（每天一份运行约束，u 连续，由 u = Σ_w z_sw·Ū^w 隐含取整），必须给热启动，用于核对。
两者的目标都是 F_K 的上界。"""
import itertools, math, time, numpy as np
import build_op as BO, ccg_solve as C, mode_lp as ML, params as PR
from milp import MILP, INF

ENUM_MAX = 50000


def choose_K(ctx, Cm, x, K, tl=300):
    """固定容量 x：从池中选 K 个模式使 inv(x) + 365·max_p Σ_s p_s·min_{w∈选中} C[s,w] 最小。
       组合数 ≤ 5 万时枚举，否则解一个不含运行约束的小 MILP。返回 (目标, 选中模式, 各日分配)。"""
    S, W = Cm.shape; P = ctx["P"]; inv = ML.inv_of(ctx, x); K = min(K, W)
    if K >= S:                                                  # 无基数约束：每天取池内最优模式（全自适应的池内上界）
        if not np.isfinite(Cm.min(1)).all(): return (np.inf, None, None)
        assign = [int(w) for w in Cm.argmin(1)]; Q = Cm.min(1)
        p = C.worst_p(Q, ctx["p0"], *ctx["th"])
        return (inv + P["DAYS"] * float(p @ Q), tuple(sorted(set(assign))), assign)
    if math.comb(W, K) <= ENUM_MAX:
        best = (np.inf, None, None)
        for comb in itertools.combinations(range(W), K):
            sub = Cm[:, comb]; Q = sub.min(1)
            if not np.isfinite(Q).all(): continue
            p = C.worst_p(Q, ctx["p0"], *ctx["th"]); f = inv + P["DAYS"] * float(p @ Q)
            if f < best[0]: best = (f, comb, [comb[i] for i in sub.argmin(1)])
        return best
    M = MILP(); fin = np.isfinite(Cm)
    v = M.var(W, 0.0, 1.0, integer=True); zz = M.var((S, W), 0.0, fin.astype(float), integer=True)
    M.con([(zz[:, w], 1.0) for w in range(W)], lo=1.0, hi=1.0)
    M.con([(zz, 1.0), (v[None, :], -1.0)], hi=0.0)
    M.row([(v, 1.0)], hi=float(K))
    Q = M.var(S, -INF, INF)
    for s in range(S):
        M.row([(Q[s], 1.0)] + [(zz[s, w], -Cm[s, w]) for w in range(W) if fin[s, w]], lo=0.0, hi=0.0)
    ML.dro_rows(M, ctx, [Q[[s]] for s in range(S)])
    sol, st, obj, g = M.solve(gap=1e-6, tl=tl)
    if sol is None: return (np.inf, None, None)
    zv = np.rint(sol[zz]).astype(int); assign = [int(np.argmax(zv[s])) for s in range(S)]
    comb = tuple(sorted(set(assign)))
    return (inv + obj, comb, assign)


def select_fast(ctx, pool, K, x0, procs=3, max_it=6, log=print, tag=""):
    """交替：成本矩阵（固定 x）选 K 个模式 → 各日分配给选中模式里成本最低者 → lp_fixed 重定容量。目标单调不增。"""
    x = dict(x0); fb = np.inf; keep = None; hist = []; t0 = time.time()
    for it in range(max_it):
        Cm = ML.cost_matrix(ctx, x, pool, procs=procs)
        f, comb, assign = choose_K(ctx, Cm, x, K)
        if comb is None: log(f"  {tag} 第 {it + 1} 轮：池内无可行组合", flush=True); break
        res = ML.lp_fixed(ctx, [pool[w] for w in assign], check=False)
        hist.append(dict(it=it + 1, f_fixed_x=f, f_lp=res["obj"], comb=comb, t=time.time() - t0))
        log(f"  {tag} 第 {it + 1} 轮：固定容量选模式 {f / 1e4:.2f} → 重定容量 {res['obj'] / 1e4:.2f} 万元，模式 {comb}，用时 {time.time() - t0:.0f} s", flush=True)
        if res["obj"] < fb - 1.0: fb, x, keep = res["obj"], res["x"], (comb, assign)
        else: break
    if keep is None: return None
    comb, assign = keep
    res = ML.lp_fixed(ctx, [pool[w] for w in assign], check=True, log=log)      # 同时充放校验（精确目标）
    return dict(x=res["x"], comb=list(comb), assign=list(assign), obj=res["obj"], Q=res["Q"], p=res["p"],
                R=res["R"], pi=res["pi"], simul=res["simul"], int_add=res["int_add"], obj_relaxed=fb,
                hist=hist, t=time.time() - t0, K=K, pool_size=len(pool))


def feasible_mask(ctx, pool, procs=3):
    """以容量上限评估各 (日, 模式) 的可行性，不可行组合在主问题中剔除。"""
    xm = dict(PR.xmax(ctx["P"]), N=ctx["N"])
    if not ctx["flags"]["tes"]: xm["TES"] = 0.0
    if not ctx["flags"]["heat"]: xm["HP"] = 0.0
    return np.isfinite(ML.cost_matrix(ctx, xm, pool, procs=procs))


def select(ctx, pool, K, start=None, mask=None, tl=600, gap=1e-4, int_add=None):
    """模式选址主问题 MILP。start = dict(assign=[各日模式序号]) 为热启动（必须给）。"""
    S, W = len(ctx["days"]), len(pool); J, T = pool[0].shape; U = np.stack(pool).astype(float)
    if mask is None: mask = np.ones((S, W), bool)
    M, X, _, Vs, _, dv = ML.build(ctx, ["free"] * S, copies=False, int_add=int_add)
    n_op = M.n
    v = M.var(W, 0.0, 1.0, integer=True); zz = M.var((S, W), 0.0, mask.astype(float), integer=True)
    M.con([(zz[:, w], 1.0) for w in range(W)], lo=1.0, hi=1.0)
    M.con([(zz, 1.0), (v[None, :], -1.0)], hi=0.0)
    M.row([(v, 1.0)], hi=float(min(K, W)))
    for s in range(S):
        u = Vs[s]["u"]
        for j in range(J):
            M.con([(u[j], 1.0)] + [(zz[s, w], -U[w, j]) for w in range(W) if U[w, j].any()], lo=0.0, hi=0.0)
    vec = None
    if start is not None:                       # 同结构的温区固定 LP 解 + v、z 取值
        Mf, Xf, _, Vf, _, _ = ML.build(ctx, [pool[w] for w in start["assign"]], copies=False, int_add=int_add)
        assert Mf.n == n_op
        sf, stf, of, _ = Mf.solve(gap=1e-6, tl=600)
        if sf is not None:
            vec = np.zeros(M.n); vec[:n_op] = sf
            for w in set(start["assign"]): vec[v[w]] = 1.0
            for s, w in enumerate(start["assign"]): vec[zz[s, w]] = 1.0
    t0 = time.time(); sol, st, obj, g = M.solve(gap=gap, tl=tl, start=vec)
    if sol is None: return dict(obj=np.inf, LB=M.dual_bound, status=st, t=time.time() - t0)
    zv = np.rint(sol[zz]).astype(int); assign = [int(np.argmax(zv[s])) for s in range(S)]
    return dict(x={c: float(sol[X[c]][0]) for c in BO.CAPS}, assign=assign, comb=sorted(set(assign)), obj=obj,
                LB=M.dual_bound, Q=np.array([float(sol[V["Q"]][0]) for V in Vs]), status=st, t=time.time() - t0)
