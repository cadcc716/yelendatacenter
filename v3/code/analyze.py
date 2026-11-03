"""
结果汇总：表E（S1–S7 容量、成本分项、运行指标、求解信息）。
口径：
  总成本 = 年化投资 + 365·Σ p*_s Q_s（p* 为该方法的最恶劣分布，SP 为 p0）；成本分项按同一 p* 加权，与总成本闭合
  能源侧年化成本 = 总成本 − GPU 购置年化值
  运行指标（PUE、ERF、满足率、碳排放、余热、弃电）按经验分布 p0 加权 × 365
  PUE = 数据中心总用电 / IT 用电，总用电不含余热热泵用电；ERF = 回收热量 / 数据中心总用电
  名义 PUE：按各场景自身模型；复核 PUE（S1、S2）：容量代入 S5 完整模型（见 table_f.py）
"""
import os, sys, pickle
import numpy as np, pandas as pd
import params as PR, build_op as BO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"S1": "DRO_S1", "S2": "DRO_S2", "S3": "DRO_S3", "S4": "DRO_S4", "S5": "DRO_S5", "S6": "SP_S6", "S7": "RO_S7"}
NAMES = {"S1": "S1 传统规划", "S2": "S2 +调频", "S3": "S3 +热耦合", "S4": "S4 +冷侧灵活性", "S5": "S5 本文", "S6": "S6 SP", "S7": "S7 RO"}


def load(site, tag):
    f = f"{ROOT}/results/{site}_{tag}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def day_items(R, P, d):
    """单日成本分项（元）与能量统计。"""
    T = P["T"]; c = R["_c"]; lam = P["CO2_PRICE"]; Vn = P["CO2_V"]
    co2 = float((lam * (1 + np.arange(Vn) * P["CO2_GROW"]) * R["eps"]).sum() - lam * R["eps_s"][0])
    return dict(
        购电=float((c["price"] * R["buy"]).sum() * P["DT"]), 售电=-float(P["C_SELL"] * R["sell"].sum() * P["DT"]),
        售热=-float(P["C_HEAT"] * R["HHP"].sum() * P["DT"]), 碳交易=co2,
        惩罚=float(P["C_R"] * R["LR"].sum() + P["C_O"] * R["O"].sum() + P["C_CUR"] * R["cur"].sum() * P["DT"]
                 + P["C_DEG"] * (R["Pch"] + R["Pdis"]).sum() * P["DT"]),
        E_total=float((R["PIT"] * (1 + c["kaux"]) + P["P0"] + R["PFC"] + R["PMC"]).sum()), E_IT=float(R["PIT"].sum()),
        cur=float(R["cur"].sum()), ren=float((R["PV"] + R["WT"] + R["cur"]).sum()), LR=float(R["LR"].sum()), AR=float(d["AR"].sum()),
        O=float(R["O"].sum()), CO2=float((P["E_GRID"] * R["buy"] - P["E_HEAT"] * R["HHP"]).sum()), Heat=float(R["HHP"].sum()),
        FCh=float((R["qFC"].sum(0) > 1.0).sum()), PIT_max=float(R["PIT"].max()), buy_max=float(R["buy"].max()))


def summarize(r, P, D, name):
    x, a = r["x"], r["a"]; p0, pw = r["p0"], r["p"]
    st = [day_items(R, P, BO.day_of(D, i)) for R, i in zip(r["R"], r["days"])]
    E0 = lambda k: 365 * sum(q * d[k] for q, d in zip(p0, st))
    Ew = lambda k: 365 * sum(q * d[k] for q, d in zip(pw, st))
    inv_gpu = a["N"] * x["N"]; inv_oth = r["inv"] - inv_gpu
    total = r["obj"]; xm = PR.xmax(P)
    row = {"场景": name, "N/卡": x["N"], "PV/MW": x["PV"] / 1e3, "WT/MW": x["WT"] / 1e3, "P_B/MW": x["PB"] / 1e3, "E_B/MWh": x["EB"] / 1e3,
           "K_FC/MW": x["FC"] / 1e3, "K_MC/MW": x["MC"] / 1e3, "E_TES/MWh": x["TES"] / 1e3, "K_HP/MW": x["HP"] / 1e3,
           "GPU投资/万元": inv_gpu / 1e4, "其他投资/万元": inv_oth / 1e4}
    for k in ("购电", "售电", "售热", "碳交易", "惩罚"): row[f"{k}/万元"] = Ew(k) / 1e4
    row.update({"能源侧年化成本/万元": (total - inv_gpu) / 1e4, "总成本/万元": total / 1e4,
                "名义PUE": E0("E_total") / E0("E_IT"), "ERF/%": 100 * E0("Heat") / E0("E_total"),
                "实时满足率/%": 100 * (1 - E0("LR") / E0("AR")), "超期量/GPU·h": E0("O"), "碳排放/t": E0("CO2"),
                "余热回收/MWh": E0("Heat") / 1e3, "弃电率/%": 100 * E0("cur") / max(E0("ren"), 1e-9), "自然冷却利用小时/h": E0("FCh"),
                "峰值IT功率/MW": max(d["PIT_max"] for d in st) / 1e3, "购电峰值/MW": max(d["buy_max"] for d in st) / 1e3,
                "迭代次数": len(r["hist"]), "最终gap": r["gap_final"], "求解时间/s": r["t"],
                "到达上限的容量": ",".join(k for k in BO.CAPS if xm[k] > 0 and x[k] >= xm[k] * (1 - 1e-4)),
                "同时充放/购售": str(r.get("simul") or "无")})
    return row


def table_e(site, P=None, mono=False):
    P = P or PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); rows = []
    for s, tag in TAGS.items():
        r = load(site, tag + ("_mono" if mono else ""))
        if r is None: continue
        P = PR.params(**r.get("params_over", {}))
        rows.append(summarize(r, P, D, NAMES[s]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    T = table_e(site); T.to_csv(f"{ROOT}/results/{site}_tableE.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(T.round(3).T.to_string())
    Tm = table_e(site, mono=True)
    if len(Tm):
        Tm.to_csv(f"{ROOT}/results/{site}_tableE_mono.csv", index=False)
        cmp = pd.DataFrame({"场景": T["场景"], "C&CG/万元": T["总成本/万元"]}).merge(
            pd.DataFrame({"场景": Tm["场景"], "单层MILP/万元": Tm["总成本/万元"]}), on="场景")
        cmp["相对差/%"] = 100 * (cmp["单层MILP/万元"] - cmp["C&CG/万元"]) / cmp["C&CG/万元"]
        print(cmp.round(4).to_string())
