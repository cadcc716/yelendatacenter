"""
K-模式预试算脚本（已在分支 claude/friendly-euler-glqw2e 的 v3/code 上跑通）。
前置改动：build_op.add_day 增加参数 u_mode / u_var / u_int（见执行方案 2.1 节）。
用法（在 v3/code 下）：
    python kmode_pilot.py T0            # 回归：现有 S5 的 12 个日模式固定 → LP
    python kmode_pilot.py CONST         # 定温基线：5 个常值温区各算一次
    python kmode_pilot.py K 1 2 3 4 6   # 成本—K 初步曲线（上界）：成本矩阵选模式 ↔ LP 重定容量 交替
说明：模式池 = 现有 S5 结果的 12 个日模式 + 5 个常值模式；结果是 F_K 的上界估计，未做同时充放校验。
"""
import os, sys, pickle, itertools, numpy as np
os.environ.setdefault("HIGHS_THREADS", "1")
import params as PR, build_op as BO, ccg_solve as C
from milp import MILP, INF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = pickle.load(open(f"{ROOT}/results/zhongwei_DRO_S5.pkl", "rb"))
P = PR.params(**(r.get("params_over") or {})); D = dict(np.load(f"{ROOT}/data/zhongwei_data.npz"))
fl = BO.flags_of("S5"); days = list(r["days"]); p0 = np.asarray(r["p0"]); th = tuple(r["theta"]); N = r["x"]["N"]
S = len(days); J = len(P["ZONES"]); T = P["T"]; Td = BO.tamb_design(D, P)
a = dict(PR.annual_cost(P), N=0.0)
modes = [np.rint(R["u"]).astype(int) for R in r["R"]]                 # 现有 S5 的 12 个日模式
const = [np.eye(J, dtype=int)[:, [j] * T] for j in range(J)]          # 5 个常值模式
pool = modes + const; W = len(pool)


def dro_rows(M, Vs):
    """与 ccg_solve.mono 相同的 DRO 对偶行。"""
    th1, thi = th; K = len(Vs)
    lam = M.var(1, -INF, INF, cost=1.0); pi = M.var(K, -INF, INF, cost=p0)
    mu = M.var(1, 0.0, INF, cost=th1); nu = M.var(K, 0.0, INF, cost=thi)
    for s, V in enumerate(Vs):
        M.row([(lam, 1.0), (pi[s], 1.0), (V["Q"], -P["DAYS"])], lo=0.0)
        M.row([(pi[s], 1.0), (mu, -1.0), (nu[s], -1.0)], hi=0.0)
        M.row([(pi[s], 1.0), (mu, 1.0), (nu[s], 1.0)], lo=0.0)


def lp_fixed(ms):
    """各日温区模式固定 → 关于容量与运行的 LP（能源侧口径，N 固定）。"""
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, integer_N=False, fixN=N); Vs = []
    for s, i in enumerate(days):
        Vs.append(BO.add_day(M, P, X, BO.day_of(D, i), fl, w=0.0, Tamb_des=Td, u_mode=ms[s]))
    dro_rows(M, Vs)
    sol, st, obj, g = M.solve(gap=1e-4, tl=600)
    if sol is None: return None, np.inf
    return {c: float(sol[X[c]][0]) for c in BO.CAPS}, obj


def day_cost(x, s, U):
    """固定容量与模式的单日 LP；不可行返回 inf。"""
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    BO.add_day(M, P, X, BO.day_of(D, days[s]), fl, w=1.0, Tamb_des=Td, u_mode=U)
    sol, st, obj, g = M.solve(gap=1e-6, tl=60)
    return obj if sol is not None else np.inf


def best_K(Cm, x, K, cands):
    """固定容量下，从候选模式中选 K 个使 DRO 目标最小（枚举）。"""
    inv = sum(a[c] * x[c] for c in BO.CAPS); best = (np.inf, None, None)
    for comb in itertools.combinations(cands, K):
        Q = Cm[:, comb].min(1)
        if not np.isfinite(Q).all(): continue
        p = C.worst_p(Q, p0, *th); f = inv + P["DAYS"] * float(p @ Q)
        if f < best[0]: best = (f, comb, Cm[:, comb].argmin(1))
    return best


if __name__ == "__main__":
    what = sys.argv[1]
    print(f"现有 S5：{r['obj'] / 1e4:.2f} 万元，下界 {r['LB'] / 1e4:.2f} 万元，N = {N:.0f}", flush=True)
    if what == "T0":
        x, f = lp_fixed(modes)
        print(f"T0：{f / 1e4:.2f} 万元", {c: round(v, 1) for c, v in x.items()}, flush=True)
    elif what == "CONST":
        for j in range(J):
            x, f = lp_fixed([const[j]] * S)
            print(f"定温 {P['ZONES'][j]}: {f / 1e4:.2f} 万元" + ("" if x is None else f"  MC {x['MC']:.0f} FC {x['FC']:.0f} TES {x['TES']:.0f} HP {x['HP']:.0f}"), flush=True)
    else:
        # 起点：单一模式中最好的一个
        f1 = [(lp_fixed([pool[w]] * S)[1], w) for w in range(W)]; fbest1, w1 = min(f1)
        x0, _ = lp_fixed([pool[w1]] * S)
        print(f"K=1: {fbest1 / 1e4:.2f} 万元，模式 {w1}", flush=True)
        for K in [int(k) for k in sys.argv[2:] if int(k) > 1]:
            xk = dict(x0); fb = np.inf; keep = None
            for it in range(4):
                Cm = np.array([[day_cost(xk, s, pool[w]) for w in range(W)] for s in range(S)])
                cands = list(range(W)) if K <= 2 else sorted(set(np.argsort(Cm, 1)[:, :3].ravel().tolist()))
                f, comb, ass = best_K(Cm, xk, K, cands)
                if comb is None: break
                assign = [comb[i] for i in ass]; xn, fn = lp_fixed([pool[w] for w in assign])
                if fn < fb - 1: fb, xk, keep = fn, xn, (comb, assign)
                else: break
            print(f"K={K}: {fb / 1e4:.2f} 万元，模式 {keep[0]}，分配 {keep[1]}，"
                  f"FC {xk['FC']:.0f} MC {xk['MC']:.0f} TES {xk['TES']:.0f} HP {xk['HP']:.0f}", flush=True)
