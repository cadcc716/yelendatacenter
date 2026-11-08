"""
结果汇总：表E（S1–S7 容量、成本分项、运行指标、求解信息）。
口径：
  两步法第二步的目标即能源侧年化成本 = 非 GPU 设备年化投资 + 365·Σ p*_s Q_s（p* 为该方法的最恶劣分布，SP 为 p0）；
  总成本 = 能源侧年化成本 + GPU 年化投资（N*）；成本分项按同一 p* 加权，与能源侧成本闭合
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
    f = f"{os.environ.get('RES_DIR', ROOT + '/results')}/{site}_{tag}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def day_items(R, P, d):
    """单日成本分项（元）与能量统计。"""
    T = P["T"]; c = R["_c"]; lam = P["CO2_PRICE"]; Vn = P["CO2_V"]
    co2 = float((lam * (1 + np.arange(Vn) * P["CO2_GROW"]) * R["eps"]).sum() - lam * R["eps_s"][0])
    return dict(
        购电=float((c["price"] * R["buy"]).sum() * P["DT"]),
        售热=-float(P["C_HEAT"] * R["HHP"].sum() * P["DT"]), 碳交易=co2,
        惩罚=float(P["C_R"] * R["LR"].sum() + P["C_O"] * R["O"].sum() + P["C_CUR"] * R["cur"].sum() * P["DT"]
                 + P["C_DEG"] * (R["Pch"] + R["Pdis"]).sum() * P["DT"]),
        E_total=float((R["PIT"] * (1 + c["kaux"]) + P["P0"] + R["PFC"] + R["PMC"]).sum()), E_IT=float(R["PIT"].sum()),
        cur=float(R["cur"].sum()), ren=float((R["PV"] + R["cur"]).sum()), LR=float(R["LR"].sum()), AR=float(d["AR"].sum()),
        O=float(R["O"].sum()), CO2=float((P["E_GRID"] * R["buy"] - P["E_HEAT"] * R["HHP"]).sum()), Heat=float(R["HHP"].sum()),
        FCh=float((R["qFC"].sum(0) > 1.0).sum()), PIT_max=float(R["PIT"].max()), buy_max=float(R["buy"].max()),
        QFC=float(R["qFC"].sum()), QMC=float(R["qMC"].sum()), QHP=float(R["qHP"].sum()), E_cool=float((R["PFC"] + R["PMC"]).sum()))


def summarize(r, P, D, name):
    x, a = r["x"], r["a"]; p0, pw = r["p0"], r["p"]
    st = [day_items(R, P, BO.day_of(D, i)) for R, i in zip(r["R"], r["days"])]
    E0 = lambda k: 365 * sum(q * d[k] for q, d in zip(p0, st))
    Ew = lambda k: 365 * sum(q * d[k] for q, d in zip(pw, st))
    inv_gpu = r["a_full"]["N"] * x["N"]; inv_oth = r["inv"]
    energy = r["obj"]; total = energy + inv_gpu; xm = PR.xmax(P)
    row = {"场景": name, "N/卡": x["N"], "PV/MW": x["PV"] / 1e3, "P_B/MW": x["PB"] / 1e3, "E_B/MWh": x["EB"] / 1e3,
           "K_FC/MW": x["FC"] / 1e3, "K_MC/MW": x["MC"] / 1e3, "E_TES/MWh": x["TES"] / 1e3, "K_HP/MW": x["HP"] / 1e3,
           "GPU投资/万元": inv_gpu / 1e4, "其他投资/万元": inv_oth / 1e4}
    for k in ("购电", "售热", "碳交易", "惩罚"): row[f"{k}/万元"] = Ew(k) / 1e4
    row.update({"能源侧年化成本/万元": energy / 1e4, "总成本/万元": total / 1e4,
                "第一步N*": r["step1"]["N"], "第一步总成本/万元": r["step1"]["obj_total"] / 1e4,
                "名义PUE": E0("E_total") / E0("E_IT"), "ERF/%": 100 * E0("Heat") / E0("E_total"),
                "实时满足率/%": 100 * (1 - E0("LR") / E0("AR")), "超期量/GPU·h": E0("O"), "碳排放/t": E0("CO2"),
                "余热回收/MWh": E0("Heat") / 1e3, "弃光率/%": 100 * E0("cur") / max(E0("ren"), 1e-9), "自然冷却利用小时/h": E0("FCh"),
                "自然冷却冷量占比/%": 100 * E0("QFC") / max(E0("QFC") + E0("QMC") + E0("QHP"), 1e-9),
                "冷却用电/MWh": E0("E_cool") / 1e3, "IT用电/MWh": E0("E_IT") / 1e3,
                "峰值IT功率/MW": max(d["PIT_max"] for d in st) / 1e3, "购电峰值/MW": max(d["buy_max"] for d in st) / 1e3,
                "迭代次数": len(r["hist"]), "最终gap": r["gap_final"], "求解时间/s": r["t"],
                "到达上限的容量": ",".join(k for k in BO.CAPS if xm[k] > 0 and x[k] >= xm[k] * (1 - 1e-4)),
                "同时充放": str(r.get("simul") or "无"), "成本闭合误差/万元": (inv_oth + Ew("购电") + Ew("售热") + Ew("碳交易") + Ew("惩罚") - energy) / 1e4})
    return row


def table_e(site, P=None, mono=False, tags=None):
    P = P or PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); rows = []
    for s, tag in (tags or TAGS).items():
        r = load(site, tag + ("_mono" if mono else "")) if not str(tag).startswith("/") else (pickle.load(open(tag, "rb")) if os.path.exists(tag) else None)
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
        cmp = pd.DataFrame({"场景": T["场景"], "C&CG/万元": T["能源侧年化成本/万元"]}).merge(
            pd.DataFrame({"场景": Tm["场景"], "单层MILP/万元": Tm["能源侧年化成本/万元"]}), on="场景")
        cmp["相对差/%"] = 100 * (cmp["单层MILP/万元"] - cmp["C&CG/万元"]) / cmp["C&CG/万元"]
        print(cmp.round(4).to_string())
