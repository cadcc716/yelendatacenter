"""
容量邻域搜索：主问题初始可行解改进（式(40)/C&CG 第二步之后调用）。
给定一组容量 x，逐典型日精确求解运行 MILP 得 Q_s(x)，按最恶劣分布得目标值 F(x) = C_inv(x) + 365·max_p Σ p_s Q_s(x)
（与式(32)在固定 x 下一致）。在 x 邻域内做坐标搜索：
  单变量移动：E_TES、K_MC、K_FC、K_HP、E_B（P_B 同比例）、P_B，步长按相对比例逐次减半；
  联合移动：E_TES 增加同时 K_MC 减少（蓄冷替代冷机）、K_HP 与 E_TES 同增。
接受严格改进，直至步长低于下限或达到评估次数上限。所得解作为最终方案；最优性仍以 MILP 的下界 LB 衡量。
"""
import os, time
import numpy as np
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP

_CTX = {}


def _init(P, D, scen):
    _CTX.update(P=P, D=D, flags=BO.flags_of(scen), Td=BO.tamb_design(D, P))


def _day(args):
    x, i = args; c = _CTX
    M = MILP(); X, _ = BO.add_capacity(M, c["P"], c["flags"], fix=x)
    V = BO.add_day(M, c["P"], X, BO.day_of(c["D"], i), c["flags"], w=1.0, Tamb_des=c["Td"])
    sol, st, obj, g = M.solve(gap=1e-4, tl=600)
    return obj


def search(P, D, scen, th, best, days, p0, a, par=None, max_eval=60, log=print, worst_p=None):
    """best: 含 x、obj 的已有解；返回改进后的解（x、Q、p、inv、obj）与评估记录。"""
    flags = BO.flags_of(scen); xm = PR.xmax(P); par = par or int(os.environ.get("PAR_LS", 4))
    vars_ = ["MC", "FC", "EB", "PB"] + (["TES"] if flags["tes"] else []) + (["HP"] if flags["heat"] else [])
    step = {"MC": 0.15, "FC": 0.15, "EB": 0.2, "PB": 0.2, "TES": 0.5, "HP": 0.3}
    floor = {"MC": 50.0, "FC": 100.0, "EB": 1000.0, "PB": 200.0, "TES": 1000.0, "HP": 100.0}
    absmin = {"TES": 5000.0, "HP": 300.0}            # 由 0 起步时的绝对步长
    cur = dict(best["x"]); fcur = best["obj"]; hist = []; t0 = time.time(); n = 0

    def feasible(x):
        if any(x[k] < -1e-6 or x[k] > xm[k] + 1e-6 for k in vars_): return False
        return x["PB"] <= P["C_B_MAX"] * x["EB"] + 1e-6

    with Pool(par, initializer=_init, initargs=(P, D, scen)) as pool:
        def F(x):
            Q = np.array(pool.map(_day, [(x, i) for i in days]))
            p = worst_p(Q, p0, *th); inv = sum(a[c] * x[c] for c in BO.CAPS)
            return inv + P["DAYS"] * float(p @ Q), Q, p, inv
        improved = True
        while n < max_eval:
            if not improved:
                if all(step[k] * max(cur[k], absmin.get(k, 0)) < floor[k] for k in vars_): break
                for k in vars_: step[k] /= 2
            improved = False
            moves = []                                  # 冷侧移动优先，储能移动最后
            if flags["tes"]:
                dt = max(step["TES"] * max(cur["TES"], absmin["TES"]), floor["TES"]); dm = max(step["MC"] * cur["MC"], floor["MC"])
                moves += [{"TES": +dt, "MC": -dm}, {"TES": +2 * dt, "MC": -2 * dm}, {"TES": -dt, "MC": +dm}]
            if flags["heat"] and flags["tes"]:
                moves += [{"HP": +max(step["HP"] * max(cur["HP"], absmin["HP"]), floor["HP"]), "TES": +max(step["TES"] * max(cur["TES"], absmin["TES"]), floor["TES"])}]
            for k in [v for v in ("MC", "TES", "HP", "FC", "EB", "PB") if v in vars_]:
                d = max(step[k] * cur[k], step[k] * absmin.get(k, 0), floor[k])
                moves += [{k: +d}, {k: -d}]
            for mv in moves:
                x = dict(cur)
                for k, d in mv.items(): x[k] = x[k] + d
                if mv.get("EB") and "PB" not in mv: x["PB"] = min(x["PB"], P["C_B_MAX"] * x["EB"])
                if not feasible(x): continue
                f, Q, p, inv = F(x); n += 1
                hist.append(dict(move=mv, obj=f))
                if f < fcur - 1.0:
                    log(f"  [邻域搜索 {scen}] {n}: {mv} → {f / 1e4:.2f} 万元（改进 {(fcur - f) / 1e4:.2f}），用时 {time.time() - t0:.0f} s", flush=True)
                    cur, fcur, keep = x, f, dict(Q=Q, p=p, inv=inv); improved = True
                    break
                if n >= max_eval: break
    if fcur < best["obj"] - 1.0:
        return dict(x=cur, obj=fcur, **keep), hist
    return None, hist
