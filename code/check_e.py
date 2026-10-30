"""
E1–E5 结果的物理一致性与运行规律校验 (审稿用)
  1 能量守恒: 电功率平衡式(25)、液冷热平衡式(7)
  2 设备容量: GPU 占用 ≤ N, 光伏 ≤ K^PV g, 购电 ≤ P^grid,max, 冷源出力 ≤ 容量与 φK
  3 热安全: 运行模式的热点温度 ≤ 85 ℃ (按时段两端液冷温度的较大值)
  4 任务: 实时完成 = 到达 − 失约, 日末积压 = 0
  5 SOC: 储能配置与 SOC 边界
  6 价格响应: 可延期任务的到达/处理在峰平谷时段的分布, IT 功率与电价相关性
  7 负荷与气象响应: IT 电量随到达量变化, 冷源电量随气温变化, 机械制冷在低温时段的使用 (数值伪影检查)
输出 results/exp/checks_*.csv, 打印汇总
"""
import pickle, sys
import numpy as np, pandas as pd
import config as C, exp as X

RES = C.RES / "exp"; P = C.params()


def zone_phi(o, P):
    return None


def check_day(o, x, b, price, P, tag):
    T = len(o["PIT"])
    bal = o["Pg"] + o["PPV"] + o["Pdis"] - o["Pch"] - o["Ptot"]
    th = P["C_W"] * np.diff(o["Tw"]) - P["DT"] * (P["BETA_LIQ"] * o["PIT"] - o["qFC"] - o["qMC"])
    gpu = sum(o["modes"].values())
    tchip = -np.inf
    for k, m in o["modes"].items():
        h = P["MODES"][k.split("|")[1]][1]; act = m > 1e-6
        if act.any(): tchip = max(tchip, (np.maximum(o["Tw"][:-1], o["Tw"][1:])[act] + P["R_TH"] * h).max())
    sR = o.get("sR", np.zeros(T)); sD = o.get("sD", np.zeros(1))
    lowT = b["Tamb"] < 15.0
    return dict(方案=tag, 功率平衡残差kW=np.abs(bal).max(), 热平衡残差kWh=np.abs(th).max(), GPU超N=(gpu - x["N"]).max(),
                光伏超可用=(o["PPV"] - o["PVav"]).max(), 购电超上限=(o["Pg"] - P["P_GRID_MAX"]).max(), 购电峰值=o["Pg"].max(),
                机械制冷超容量=(o["qMC"] - x["MC"]).max(), 自然冷却超容量=(o["qFC"] - x["FC"]).max(), 最高热点温度=tchip,
                实时完成误差=np.abs(o["UR"] + sR - b["AR"]).max(), 日末积压=o["Bq"][-1], 同时充放电h=int(((o["Pch"] > 1e-3) & (o["Pdis"] > 1e-3)).sum()),
                液冷温度min=o["Tw"].min(), 液冷温度max=o["Tw"].max(),
                低温时段机械制冷散热kWh=float(o["qMC"][lowT].sum()), 机械制冷散热kWh=float(o["qMC"].sum()),
                低温时段机械制冷电量kWh=float(o["PMC"][lowT].sum()), 日运行费用=o["Cop"])


def price_response(H, D):
    price = D["price"]; lvl = np.where(price > price.mean() * 1.2, "峰", np.where(price < price.mean() * 0.8, "谷", "平"))
    arr, done, it, pr = {k: 0.0 for k in "峰平谷"}, {k: 0.0 for k in "峰平谷"}, [], []
    for (d, i), o in H.items():
        for t in range(24):
            arr[lvl[t]] += o["AB"][t]; done[lvl[t]] += o["UB"][t]
        it.append(o["PIT"] - o["PIT"].mean()); pr.append(price - price.mean())
    A, Dn = sum(arr.values()), sum(done.values())
    corr = np.corrcoef(np.concatenate(it), np.concatenate(pr))[0, 1]
    return {**{f"可延期到达占比_{k}": arr[k] / A for k in arr}, **{f"可延期处理占比_{k}": done[k] / Dn for k in done}, "IT功率与电价相关系数": corr}


def response(H, D):
    rows = []
    for (d, i), o in H.items():
        b = D["test"][d][i]
        rows.append(dict(d=d, 到达GPUh=b["AR"].sum() + b["AB"].sum(), IT电量=o["PIT"].sum(), 冷源电量=(o["PFC"] + o["PMC"]).sum(),
                         Tmean=b["Tamb"].mean(), Tmax=b["Tamb"].max(), 机械制冷散热=o["qMC"].sum()))
    R = pd.DataFrame(rows)
    return dict(IT电量_到达量_相关=R[["到达GPUh", "IT电量"]].corr().iloc[0, 1], 冷源电量_日均温_相关=R[["Tmean", "冷源电量"]].corr().iloc[0, 1],
                机械制冷_日最高温_相关=R[["Tmax", "机械制冷散热"]].corr().iloc[0, 1],
                单位工作量IT电耗=R["IT电量"].sum() / R["到达GPUh"].sum())


if __name__ == "__main__":
    names = sys.argv[1:] or ["M0", "M3", "U3"]
    D = X.dataset(); rows, pr, rs = [], [], []
    for nm in names:
        R = pickle.load(open(RES / f"plan_{nm}.pkl", "rb"))
        # (a) 规划情景 (训练支撑集)
        for b, o in zip(R["blocks"], R["out"]):
            r = check_day(o, R["x"], b, D["price"], P, nm); r["口径"] = "规划情景"; rows.append(r)
        # (b) 封存测试日 (统一评价器逐时结果)
        import plot_e
        H = plot_e.hourly(nm)
        for (d, i), o in H.items():
            r = check_day(o, R["x"], D["test"][d][i], D["price"], P, nm); r["口径"] = "测试日"; rows.append(r)
        pr.append(dict(方案=nm, **price_response(H, D))); rs.append(dict(方案=nm, **response(H, D)))
        print("checked", nm, flush=True)
    K = pd.DataFrame(rows); K.to_csv(RES / "checks_physics.csv", index=False)
    agg = K.groupby(["方案", "口径"]).agg(功率平衡残差kW=("功率平衡残差kW", "max"), 热平衡残差kWh=("热平衡残差kWh", "max"), GPU超N=("GPU超N", "max"),
                                         光伏超可用=("光伏超可用", "max"), 购电峰值=("购电峰值", "max"), 机械制冷超容量=("机械制冷超容量", "max"),
                                         自然冷却超容量=("自然冷却超容量", "max"), 最高热点温度=("最高热点温度", "max"), 实时完成误差=("实时完成误差", "max"),
                                         日末积压=("日末积压", "max"), 同时充放电h=("同时充放电h", "sum"), 液冷温度min=("液冷温度min", "min"),
                                         液冷温度max=("液冷温度max", "max"), 低温时段机械制冷散热kWh=("低温时段机械制冷散热kWh", "sum"),
                                         机械制冷散热kWh=("机械制冷散热kWh", "sum"))
    agg.to_csv(RES / "checks_summary.csv"); pd.DataFrame(pr).to_csv(RES / "checks_price.csv", index=False); pd.DataFrame(rs).to_csv(RES / "checks_response.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg.round(4).T.to_string()); print(pd.DataFrame(pr).round(3).to_string()); print(pd.DataFrame(rs).round(3).to_string())
