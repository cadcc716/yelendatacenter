"""
结果汇总：表E（S1–S7 指标）、表F（S2 容量代入 S3 完整模型）、验收项 C1–C7、C9–C11。
指标口径：
  能源侧年化成本 = 总成本 − GPU 购置年化值；总成本 = 年化投资 + 365·Σ p*_s Q_s（p* 为该方法的最恶劣分布，SP 为 p0）
  其余年度指标按经验分布 p0 加权（典型日 × 365）
  PUE = 园区总用电 / IT 用电，总用电不含余热热泵用电（热泵服务于对外供热，按能源回用另计）
"""
import os, sys, pickle, json
import numpy as np, pandas as pd
import params as PR, build_op as BO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"S1": "DRO_S1", "S2": "DRO_S2", "S3": "DRO_S3", "S4": "DRO_S4", "S5": "DRO_S5", "S6": "SP_S6", "S7": "RO_S7"}
NAMES = {"S1": "S1 传统规划", "S2": "S2 +调频", "S3": "S3 +热耦合", "S4": "S4 +冷侧灵活性", "S5": "S5 本文", "S6": "S6 SP", "S7": "S7 RO"}


def load(site, s):
    f = f"{ROOT}/results/{site}_{TAGS[s]}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def day_stats(R, P, d):
    T = P["T"]; c = R["_c"]
    load_ = R["PIT"] * (1 + c["kaux"]) + P["P0"] + R["PFC"] + R["PMC"]
    m = np.zeros((5, T))
    for a, k in enumerate(P["K_R"]): m[k] += R["mR"][a].sum(0)
    for a, k in enumerate(P["K_B"]): m[k] += R["mB"][a].sum(0)
    return dict(E_total=load_.sum(), E_IT=R["PIT"].sum(), E_HP=R["PHP"].sum(), buy=R["buy"].sum(), sell=R["sell"].sum(),
                cur=R["cur"].sum(), ren=(R["PV"] + R["WT"] + R["cur"]).sum(), LR=R["LR"].sum(), AR=d["AR"].sum(), O=R["O"].sum(),
                UB=R["UB"].sum(), CO2=(P["E_GRID"] * R["buy"] - P["E_HEAT"] * R["HHP"]).sum(), Heat=R["HHP"].sum(),
                FCh=(R["qFC"].sum(0) > 1e-3 * max(1.0, R["qFC"].sum(0).max())).sum(), FC=R["qFC"].sum(), MC=R["qMC"].sum(),
                PIT_max=R["PIT"].max(), buy_max=R["buy"].max(), k1=m[0], act=m.sum(0), Tw=R["Tw"][:T],
                simul_B=np.minimum(R["Pch"], R["Pdis"]).max(), simul_G=np.minimum(R["buy"], R["sell"]).max(),
                simul_T=np.minimum(R["qch"], R["qdis"]).max())


def table_e(site, P=None):
    P = P or PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); rows = []
    for s in TAGS:
        r = load(site, s)
        if r is None: continue
        x, a = r["x"], r["a"]; p0 = r["p0"]; days = r["days"]
        st = [day_stats(R, P, BO.day_of(D, i)) for R, i in zip(r["R"], days)]
        E = lambda k: 365 * sum(q * d[k] for q, d in zip(p0, st))
        total = r["inv"] + 365 * float(r["p"] @ r["Q"])
        rows.append({"场景": NAMES[s], "GPU/卡": x["N"], "光伏/MW": x["PV"] / 1e3, "风电/MW": x["WT"] / 1e3, "储能/MWh": x["EB"] / 1e3,
                     "储能功率/MW": x["PB"] / 1e3, "自然冷却/MW": x["FC"] / 1e3, "机械制冷/MW": x["MC"] / 1e3, "蓄冷/MWh": x["TES"] / 1e3,
                     "热泵/MW": x["HP"] / 1e3, "PUE": E("E_total") / E("E_IT"), "能源侧年化成本/万元": (total - a["N"] * x["N"]) / 1e4,
                     "总成本/万元": total / 1e4, "期望能源侧成本(p0)/万元": (r["inv"] - a["N"] * x["N"] + 365 * float(p0 @ r["Q"])) / 1e4,
                     "弃电率/%": 100 * E("cur") / max(E("ren"), 1e-9), "实时任务满足率/%": 100 * (1 - E("LR") / E("AR")),
                     "超期量/GPU·h": E("O"), "碳排放/t": E("CO2"), "余热回收/MWh": E("Heat") / 1e3, "自然冷却利用小时/h": E("FCh"),
                     "峰值IT功率/MW": max(d["PIT_max"] for d in st) / 1e3, "购电峰值/MW": max(d["buy_max"] for d in st) / 1e3,
                     "迭代次数": len(r["hist"]), "最终gap": r["hist"][-1]["gap"], "求解时间/s": r["hist"][-1]["t"],
                     "同时充放/kW": max(d["simul_B"] for d in st), "同时购售/kW": max(d["simul_G"] for d in st), "蓄冷同时充放/kW": max(d["simul_T"] for d in st)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    T = table_e(site)
    T.to_csv(f"{ROOT}/results/{site}_tableE.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(T.round(3).T.to_string())
