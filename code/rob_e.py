"""
审稿第 1–3 项的稳健性汇总 (统一评价器, 封存测试日年化)
  rob_sens.csv : 关键物理参数一次一变 (R_th, T_chip_max, P_OFF, ρ): M3 冷源配置、M3 相对 M0 的能源相关费用差
  rob_fan.csv  : 风机变工况模型下的 E1 (M0–M3, M0-设计工况): 费用、冷却电量、PUE、冷源电量与气温相关性、交互项
  rob_long.csv : BurstGPT 188 日负荷 (负荷与气象独立) 下的 M0 / M0-设计工况 / M3 / 软约束 SP / 软约束 U3
"""
import pickle
import numpy as np, pandas as pd
import config as C, exp as X
import risk_e as RK

RES = C.RES / "exp"; W = 1e4


def row(nm):
    f = RES / f"row_{nm}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def coolcorr(nm):
    f = RES / f"hf_{nm}.csv"
    if not f.exists(): return np.nan
    E = pd.read_csv(f)
    return float(np.corrcoef(E["Tmax"], E["冷却电量"])[0, 1]) if E["冷却电量"].std() > 0 else np.nan


def sens():
    out = []
    base = [("基准", "M0", "M3")] + [(tg, f"SENS-{tg}-M0", f"SENS-{tg}-M3") for tg in X.SENS]
    lab = {"基准": "基准 (R_th=57.1 K/kW, 85 ℃, 0.03 kW, ρ=0.2)", "Rth0.8": "热阻 ×0.8", "Rth1.2": "热阻 ×1.2", "Tc83": "芯片上限 83 ℃",
           "Tc87": "芯片上限 87 ℃", "Poff0.08": "休眠功率 0.08 kW", "Poff0.15": "休眠功率 0.15 kW", "rho0.1": "热备用 10%", "rho0.4": "热备用 40%"}
    for tg, m0, m3 in base:
        a, b = row(m0), row(m3)
        if a is None or b is None: continue
        P = C.params(X.PLANS[m3].get("cfg"))
        safe = {k: P["T_CHIP_MAX"] - P["R_TH"] * P["MODES"][k][1] for k in P["MODES"]}
        out.append(dict(参数=lab[tg], 标称安全温度=safe["k1_标称"], 浅锁频安全温度=safe["k2_浅锁频"], M3自然冷却=b["FC"], M3机械制冷=abs(max(b["MC"], 0)),
                        M0能源相关费用=a["复核能源相关费用"] / W, M3能源相关费用=b["复核能源相关费用"] / W,
                        M3较M0=(b["复核能源相关费用"] - a["复核能源相关费用"]) / W, M3较M0百分比=(b["复核能源相关费用"] / a["复核能源相关费用"] - 1) * 100,
                        M3冷却失约=b["实时失约_冷却热安全受限"], M3间隙=b["间隙"] / W))
    return pd.DataFrame(out)


def fan():
    out = []
    for grp, pre in (("常数风机电耗", ""), ("风机变工况", "FAN-")):
        for m in ("M0", "M0-设计工况", "M1", "M2", "M3"):
            r = row(pre + m)
            if r is None: continue
            out.append(dict(风机模型=grp, 方案=m, 自然冷却=r["FC"], 机械制冷=abs(max(r["MC"], 0)), 光伏=r["PV"], 能源相关费用=r["复核能源相关费用"] / W,
                            冷却电量MWh=r["复核冷却电量MWh"], PUE=r["复核PUE"], 冷源电量与日最高温相关=coolcorr(pre + m), 间隙=r["间隙"] / W))
    T = pd.DataFrame(out)
    for grp in T["风机模型"].unique():
        t = T[T["风机模型"] == grp].set_index("方案")["能源相关费用"]
        if all(k in t for k in ("M0", "M1", "M2", "M3")):
            T.loc[T["风机模型"] == grp, "交互项I_GT"] = t["M1"] + t["M2"] - t["M0"] - t["M3"]
    return T


def long():
    D = X.dataset("long"); nd = D["n_d"]; out = []
    for nm, lab in (("L-M0", "M0 静态基准"), ("L-M0-设计工况", "M0-设计工况"), ("L-M3", "M3 完整耦合(硬约束)"),
                    ("L-R100-SP", "软约束 SP (c_R=100)"), ("L-R100-U3", "软约束 U3 (c_R=100)")):
        r = RK.summary(nm, 100, nd); rw = row(nm)
        if r is None or rw is None: continue
        out.append(dict(方案=lab, GPU=rw["N"], 光伏=rw["PV"], 自然冷却=rw["FC"], 机械制冷=abs(max(rw["MC"], 0)), 能源相关费用=rw["复核能源相关费用"] / W,
                        实时SLA=rw["实时SLA"] * 100, 年化冷却受限失约=r["其中冷却受限"], 风险调整费用=r["风险调整费用"], 间隙=rw["间隙"] / W))
    R0 = pickle.load(open(RES / "plan_L-M3.pkl", "rb")) if (RES / "plan_L-M3.pkl").exists() else None
    for f in sorted(RES.glob("hf_L-M3-FC*.csv")):           # M3 容量 + 自然冷却余量 (仅改变 K^FC, 统一评价器复核)
        fc = float(f.stem.split("FC")[-1]); E = pd.read_csv(f); x = dict(R0["x"], FC=fc); a = R0["a"]
        inv = sum(a[k] * x[k] for k in x) - a["N"] * x["N"]
        op = RK.annual(E, E["日运行费用"].fillna(0).values, nd); sR = RK.annual(E, E["实时失约"].fillna(0).values, nd)
        gl = RK.annual(E, E["实时失约_GPU台数受限"].fillna(0).values, nd); sD = RK.annual(E, E["延期失约"].fillna(0).values, nd)
        out.append(dict(方案=f"M3 + 自然冷却余量 (FC={fc:.0f})", GPU=x["N"], 光伏=x["PV"], 自然冷却=fc, 机械制冷=0.0, 能源相关费用=(inv + op) / W,
                        实时SLA=(1 - E["实时失约"].sum() / E["实时到达"].sum()) * 100, 年化冷却受限失约=sR - gl,
                        风险调整费用=(inv + op + 100 * sR + 20 * sD) / W, 间隙=np.nan))
    B = None
    if RK.load("L-R100-U3") is not None and RK.load("L-R100-SP") is not None:
        B = pd.DataFrame([RK.boot("L-R100-U3", "L-R100-SP", 100, nd), RK.boot("L-M3", "L-M0", 100, nd)])
    return pd.DataFrame(out), B


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    S = sens(); S.to_csv(RES / "rob_sens.csv", index=False); print(S.round(2).to_string())
    F = fan(); F.to_csv(RES / "rob_fan.csv", index=False); print(F.round(3).to_string())
    Lg, B = long(); Lg.to_csv(RES / "rob_long.csv", index=False); print(Lg.round(2).to_string())
    if B is not None: B.to_csv(RES / "rob_long_boot.csv", index=False); print(B.round(2).to_string())
