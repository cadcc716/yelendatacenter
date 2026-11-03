"""第二步：固定一组人工容量，求解夏季、冬季典型日（S5 完整模型），核对 C10、C11 与调试检查表单日项。"""
import sys, time
import numpy as np
import params as PR, build_op as BO
from milp import MILP

X_TEST = dict(N=5500, PV=10000, WT=5000, PB=10000, EB=20000, FC=9000, MC=8000, TES=20000, HP=3000)


def solve_day(D, i, scen="S5", x=X_TEST, P=None, verbose=False):
    P = P or PR.params(); fl = BO.flags_of(scen)
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, Tamb_des=BO.tamb_design(D, P))
    t0 = time.time(); sol, st, obj, gap = M.solve(gap=1e-4, tl=600, verbose=verbose)
    return BO.extract(sol, V), st, obj, time.time() - t0, P


def report(R, P, D, i, tag):
    T = P["T"]; KR, KB = P["K_R"], P["K_B"]
    m = np.zeros((5, T))
    for a, k in enumerate(KR): m[k] += R["mR"][a].sum(0)
    for a, k in enumerate(KB): m[k] += R["mB"][a].sum(0)
    act = m.sum(0); zone = R["u"].argmax(0)
    k1 = np.where(act > 1, m[0] / np.maximum(act, 1e-9), np.nan)
    noon = slice(12, 16); night = slice(0, 6)
    UB = R["UB"]; val = np.isin(np.arange(T), P["VALLEY_H"])
    print(f"--- {tag}: 日期 {D['dates'][i]}, 气温 {D['Tamb'][i].min():.1f}–{D['Tamb'][i].max():.1f} ℃")
    print("  液冷温度 T_w:", np.round(R["Tw"][:T], 1))
    print("  温区 j     :", zone + 1)
    print("  工作GPU    :", np.round(act).astype(int))
    print("  k1占比     :", np.round(k1, 2))
    for k in range(5): print(f"  档位{k + 1}GPU  :", np.round(m[k]).astype(int))
    print(f"  C10 12–16时 k1 占比 = {np.nanmean(k1[noon]):.2f} (<0.30), 0–6时 = {np.nanmean(k1[night]):.2f} (>0.60)")
    print(f"  C11 谷时段可延期任务处理占比 = {UB[val].sum() / UB.sum():.2f} (≥0.40)；谷时段到达占比 = {D['AB'][i][val].sum() / D['AB'][i].sum():.2f}")
    print(f"  T_w≥45 ℃ 时段占比 = {(R['Tw'][:T] >= 45 - 1e-6).mean():.2f}；实时切除 {R['LR'].sum():.0f} GPU·h，超期 {R['O'].sum():.0f}")
    print("  q_FC:", np.round(R["qFC"].sum(0)).astype(int))
    print("  q_MC:", np.round(R["qMC"].sum(0)).astype(int))
    print("  q_HP:", np.round(R["qHP"].sum(0)).astype(int), " H_dem:", np.round(D["H"][i]).astype(int))
    print("  蓄冷 S:", np.round(R["S"][:T]).astype(int), " ch:", np.round(R["qch"]).astype(int), " dis:", np.round(R["qdis"]).astype(int))
    print("  储能 SB:", np.round(R["SB"][:T]).astype(int))
    print("  购电:", np.round(R["buy"]).astype(int), " 售电:", np.round(R["sell"]).astype(int))
    print("  可延期 到达:", np.round(D["AB"][i]).astype(int), "\n         处理:", np.round(UB).astype(int))
    print(f"  碳: 区间 {np.round(R['eps'], 1)} 富余出售 {R['eps_s'][0]:.1f} t；日运行成本 {R['Q'][0] / 1e4:.2f} 万元")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    D = dict(np.load(f"../data/{site}_data.npz"))
    med = D["med"]; Tmax = D["Tamb"][med].max(1)
    iS = med[Tmax.argmax()]; iW = med[Tmax.argmin()]
    over = dict(a.split("=") for a in sys.argv[2:])
    P0 = PR.params(**{k: (eval(v)) for k, v in over.items()})
    for i, tag in ((iS, "夏季典型日"), (iW, "冬季典型日")):
        R, st, obj, dt, P = solve_day(D, i, P=P0)
        print(f"[{tag}] 状态 {st}, 目标 {obj:.0f}, 用时 {dt:.1f} s")
        report(R, P, D, i, tag)
