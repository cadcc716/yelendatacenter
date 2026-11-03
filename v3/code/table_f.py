"""
表F：把 S1、S2 的容量代入 S5 完整模型（热安全门控 + 漏电 + 气象冷却 + 液冷热惯性）逐典型日运行，统计
  实时缺额、超期量（按 p0 年化），以及 S1/S2 自身名义运行中的热越限时长（所用档位不在当前温区的安全集合内，
  芯片热点温度 T̄_j + R_th·h_k 超过 85 ℃）与最大超温幅度。同时给出 S1、S2 的复核 PUE。
用法：python table_f.py <site>
"""
import os, sys, pickle
os.environ.setdefault("HIGHS_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(args):
    site, x, s, i = args
    P = PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); fl = BO.flags_of("S5")
    x = dict(x, TES=0.0, HP=0.0)
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, w=1.0, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-4, tl=900)
    R = BO.extract(sol, V); c = R["_c"]
    return dict(s=s, day=i, Q=obj, LR=R["LR"].sum(), LR_h=int((R["LR"] > 1).sum()), AR=D["AR"][i].sum(), O=R["O"].sum(),
                O_h=int((R["O"] > 1).sum()), E_tot=float((R["PIT"] * (1 + c["kaux"]) + P["P0"] + R["PFC"] + R["PMC"]).sum()),
                E_IT=float(R["PIT"].sum()), MC_max=float(R["qMC"].sum(0).max()), status=st)


def nominal_violation(r, P):
    """S1/S2 名义运行：各典型日逐时检查所用档位是否在当前温区的安全集合内。"""
    Z = PR.zone_consts(P); h = np.array([g[3] for g in P["GEARS"]]); hrs = []; dT = []
    for R in r["R"]:
        zone = R["u"].argmax(0); m = np.zeros((5, P["T"]))
        for a, k in enumerate(P["K_R"]): m[k] += R["mR"][a][zone, np.arange(P["T"])]
        for a, k in enumerate(P["K_B"]): m[k] += R["mB"][a][zone, np.arange(P["T"])]
        Tchip = Z["hi"][zone][None, :] + P["R_TH"] * h[:, None]          # 各档位芯片热点温度（温区上界）
        over = (m > 1) & (Tchip > P["T_CHIP_MAX"] + 1e-9)
        hrs.append(int(over.any(0).sum())); dT.append(float(((Tchip - P["T_CHIP_MAX"]) * over).max()))
    return np.array(hrs), np.array(dT)


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    P = PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); p0 = D["p0"]; rows = []; detail = []
    for tag in ("DRO_S1", "DRO_S2"):
        r = pickle.load(open(f"{ROOT}/results/{site}_{tag}.pkl", "rb"))
        with Pool(int(os.environ.get("PAR", 2))) as pool: res = pool.map(one, [(site, r["x"], s, i) for s, i in enumerate(r["days"])])
        hrs, dT = nominal_violation(r, P)
        E = lambda k: 365 * sum(q * d[k] for q, d in zip(p0, res))
        imax = int(np.argmax(D["Tamb"][r["days"]].max(1)))                         # 最高气温典型日
        iw = int(np.argmax([d["MC_max"] + 1e3 * d["LR_h"] for d in res]))         # 冷却最紧张（缺额最多）典型日
        rows.append({"方案": tag[-2:], "复核PUE": E("E_tot") / E("E_IT"), "年实时缺额/GPU·h": E("LR"), "实时满足率/%": 100 * (1 - E("LR") / E("AR")),
                     "年实时缺额时长/h": E("LR_h"), "年超期量/GPU·h": E("O"), "年超期时长/h": E("O_h"),
                     "名义运行热越限时长/h": 365 * float(p0 @ hrs), "名义运行最大超温/K": float(dT.max()),
                     "夏季典型日缺额时长/h": res[imax]["LR_h"], "夏季典型日超期时长/h": res[imax]["O_h"],
                     "夏季典型日名义热越限/h": int(hrs[imax]), "夏季典型日日期": str(D["dates"][r["days"][imax]]),
                     "最不利典型日日期": str(D["dates"][r["days"][iw]]), "最不利日缺额时长/h": res[iw]["LR_h"],
                     "最不利日缺额/GPU·h": res[iw]["LR"], "最不利日名义热越限/h": int(hrs[iw])})
        for d in res: detail.append(dict(方案=tag[-2:], 日期=str(D["dates"][d["day"]]), **{k: v for k, v in d.items() if k not in ("s", "day")}))
    T = pd.DataFrame(rows); T.to_csv(f"{ROOT}/results/{site}_tableF.csv", index=False)
    pd.DataFrame(detail).to_csv(f"{ROOT}/results/{site}_tableF_days.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(3).T.to_string())
