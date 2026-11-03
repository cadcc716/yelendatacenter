"""
数据准备（执行方案 5.1 节、式(30)(31)）
  气象：Open-Meteo 历史逐时 JSON（2023 年），换算为光伏、风电单位出力系数
  算力负荷：Azure LLM 推理 trace（2024-05 一周）→ 实时任务日内形状与工作日/周末系数
            Alibaba cluster-trace-gpu-v2020（PAI 任务表）→ 可延期任务逐日到达形状
  热负荷：采暖季（10-15 至次年 04-15）随日均气温线性变化，峰值 3 MW_th；非采暖季 0.3 MW_th 生活热水
  典型日：6 类日轨迹联合归一化，k-medoids 聚为 K 个典型日，输出 ξ_s、p0、θ1、θ∞

用法：python data_prep.py <site>      site ∈ SITES
原始 trace 的逐时汇总（v3/data/trace/*.csv）由 aggregate_traces() 从原始文件生成。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import params as PR

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "v3"; OUT = V3 / "data"; TRACE = OUT / "trace"
SITES = {   # 名称: (纬度, 经度, 气象文件, 年份)
    "ulanqab": (41.03, 113.13, ROOT / "data/raw/ulanqab_2023.json", 2023),
    "guian": (26.40, 106.39, ROOT / "data/raw/guian_2023.json", 2023),
    "shaoguan": (24.81, 113.60, ROOT / "data/raw/shaoguan_2023.json", 2023),
    "zhongwei": (37.50, 105.22, ROOT / "data/raw/zhongwei_era5.json", 2023),   # 调试替代站点（同为国家算力枢纽）
}
HUB = 100.0            # 风机轮毂高度 m
SHEAR = 1 / 7          # 风切变指数（仅在缺少 100 m 风速时由 10 m 外推）


# ---------------------------------------------------------------- trace 汇总
def aggregate_traces(azure_csv, ali_csv):
    """从原始 trace 生成逐时汇总；仅在重新生成时调用。"""
    parts = []
    for ch in pd.read_csv(azure_csv, chunksize=5_000_000):
        t = pd.to_datetime(ch["TIMESTAMP"], format="mixed", utc=True).dt.floor("h")
        parts.append(pd.DataFrame({"t": t, "n": 1, "ctx": ch["ContextTokens"], "gen": ch["GeneratedTokens"]}).groupby("t").sum())
    pd.concat(parts).groupby(level=0).sum().to_csv(TRACE / "azure_hourly.csv")
    cols = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
    T = pd.read_csv(ali_csv, header=None, names=cols)
    T = T[(T.plan_gpu > 0) & T.start_time.notna() & T.end_time.notna()]
    T["gh"] = T.inst_num * T.plan_gpu / 100 * (T.end_time - T.start_time) / 3600      # 任务 GPU·h，按提交时刻计入到达
    T["h"] = (T.start_time // 3600).astype(int)
    H = T.groupby("h")["gh"].sum(); H = H.reindex(range(H.index.min(), H.index.max() + 1), fill_value=0)
    H.to_csv(TRACE / "ali_hourly_gh.csv")


def azure_shapes():
    """实时任务：按推理工作量 w = 0.1·输入 token + 输出 token（预填充单 token 计算量约为解码的 1/10）。
    trace 时间为 UTC，负荷高峰对应美国东部工作时段，按 America/New_York 换算为用户本地时间。"""
    A = pd.read_csv(TRACE / "azure_hourly.csv", index_col=0, parse_dates=True)
    A.index = A.index.tz_convert("America/New_York")
    w = A["ctx"] * 0.1 + A["gen"]
    d = pd.DataFrame({"w": w.values, "date": A.index.date, "h": A.index.hour, "wd": A.index.dayofweek})
    d = d.groupby(["date", "h", "wd"], as_index=False)["w"].sum()
    full = d.groupby("date")["h"].transform("count") == 24
    d = d[full]
    tot = d.groupby(["date", "wd"])["w"].sum().reset_index()
    wk = tot[tot.wd < 5]["w"].mean(); we = tot[tot.wd >= 5]["w"].mean()
    def shape(mask):
        s = d[mask].groupby("h")["w"].mean().reindex(range(24)).values
        return s / s.mean()
    m = (wk + we * 2 / 5) / (1 + 2 / 5)
    return dict(wd=shape(d.wd < 5), we=shape(d.wd >= 5), f_wd=wk / m, f_we=we / m)


def ali_days():
    """可延期任务：按任务提交时刻的 GPU·h 统计到达量，时间戳偏移 +8 h 对齐北京时间；
    每日曲线做 3 h 循环滑动平均，返回 (n_days, 24) 归一化日曲线（全体均值为 1）。"""
    H = pd.read_csv(TRACE / "ali_hourly_gh.csv", index_col=0).iloc[:, 0]
    h = H.index.values + 8
    s = pd.Series(H.values, index=h)
    d0 = int(np.ceil(h.min() / 24)); d1 = int(h.max() // 24)
    X = np.array([s.reindex(range(24 * d, 24 * d + 24), fill_value=0).values for d in range(d0, d1)], float)
    X = (np.roll(X, 1, 1) + X + np.roll(X, -1, 1)) / 3
    day = X.sum(1); keep = day > 0.2 * np.median(day)          # 剔除采集缺失日
    X = X[keep]; f = X.sum(1) / X.sum(1).mean()
    X = X * np.clip(f, 0.6, 1.4)[:, None] / f[:, None]          # 日总量波动截断在 ±40%
    return X / X.mean()


# ---------------------------------------------------------------- 气象与风光
def load_weather(site):
    lat, lon, f, yr = SITES[site]
    if not f.exists(): raise FileNotFoundError(f"缺少气象文件 {f}")
    d = json.load(open(f)); h = d["hourly"]; u = d.get("hourly_units", {})
    W = pd.DataFrame(h); W["time"] = pd.to_datetime(W["time"])
    W = W[W.time.dt.year == yr].set_index("time")
    if "Asia/Shanghai" not in d.get("timezone", "") and d.get("utc_offset_seconds", 0) != 28800:
        W.index = W.index.tz_localize("UTC").tz_convert("Asia/Shanghai").tz_localize(None)
    for c in [c for c in W.columns if c.startswith("wind_speed")]:
        if u.get(c, "m/s").startswith("km"): W[c] = W[c] / 3.6
    W = W[~W.index.duplicated()].reindex(pd.date_range(f"{yr}-01-01", f"{yr}-12-31 23:00", freq="h")).interpolate(limit_direction="both")
    if len(W) == 8784: W = W[~((W.index.month == 2) & (W.index.day == 29))]
    return W


def pv_output(W, lat, lon):
    """固定倾角（=纬度、正南）组件，Hay-Davies 天空散射模型；电池温度按 NOCT 法，
    温度系数 −0.37 %/K，系统综合效率 0.86。返回单位装机出力系数。"""
    import pvlib
    t = W.index.tz_localize("Asia/Shanghai")
    sp = pvlib.solarposition.get_solarposition(t - pd.Timedelta("30min"), lat, lon)   # 时段中点（Open-Meteo 为前一小时平均）
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(t))
    poa = pvlib.irradiance.get_total_irradiance(abs(lat), 180, sp["apparent_zenith"].values, sp["azimuth"].values,
                                                W["direct_normal_irradiance"].values, W["shortwave_radiation"].values,
                                                W["diffuse_radiation"].values, dni_extra=dni_extra, model="haydavies")
    G = np.nan_to_num(np.asarray(poa["poa_global"], float)).clip(0)
    Tc = W["temperature_2m"].values + (45 - 20) / 800 * G
    return np.clip(G / 1000 * (1 - 0.0037 * (Tc - 25)) * 0.86, 0, 1)


def wind_output(W):
    """轮毂 100 m 风速（缺失时由 10 m 风速按指数律外推）；通用 2–3 MW 级功率曲线：
    切入 3 m/s、额定 11 m/s、切出 25 m/s，切入至额定间按三次方。"""
    v = W["wind_speed_100m"].values if "wind_speed_100m" in W else W["wind_speed_10m"].values * (HUB / 10) ** SHEAR
    g = np.where(v < 3, 0, np.where(v < 11, (v ** 3 - 27) / (11 ** 3 - 27), np.where(v <= 25, 1, 0)))
    return g.clip(0, 1), v


# ---------------------------------------------------------------- 负荷与热负荷
def heat_demand(P, Tamb_day, dates):
    """采暖季热负荷 ∝ (18 − 日均气温)，叠加生活热水；日内形状取居民/办公典型曲线。"""
    heat_shape = np.array([1.10, 1.12, 1.14, 1.15, 1.15, 1.18, 1.20, 1.15, 1.02, 0.92, 0.85, 0.80,
                           0.78, 0.78, 0.80, 0.84, 0.90, 0.98, 1.05, 1.08, 1.08, 1.06, 1.05, 1.08])
    dhw_shape = np.array([0.35, 0.30, 0.30, 0.30, 0.40, 0.70, 1.30, 1.60, 1.30, 1.00, 0.90, 0.90,
                          1.00, 0.90, 0.80, 0.80, 0.90, 1.10, 1.40, 1.70, 1.80, 1.50, 1.00, 0.60])
    heat_shape = heat_shape / heat_shape.max(); dhw_shape = dhw_shape / dhw_shape.max()
    md = np.array([d.month * 100 + d.day for d in dates])
    season = (md >= 1015) | (md <= 415)
    Tm = Tamb_day.mean(1)
    hdd = np.clip(18 - Tm, 0, None) * season
    H = P["H_SUMMER"] * dhw_shape[None, :] + (hdd / max(hdd.max(), 1e-9))[:, None] * heat_shape[None, :]
    base = P["H_SUMMER"] * dhw_shape[None, :]; sh = H - base
    lo, hi = 0.0, 10 * P["H_PEAK"] / max(sh.max(), 1e-9)
    for _ in range(60):                       # 二分使全年峰值恰为 H_PEAK
        k = (lo + hi) / 2
        lo, hi = (k, hi) if (base + k * sh).max() < P["H_PEAK"] else (lo, k)
    return base + k * sh, season


def compute_load(P, dates):
    """实时任务 A_R 与可延期任务 A_B（单位 GPU·h/h，按满频等效）。
    叠加工作日/周末系数与季节缩放（±5%，7 月中旬最高）；全年日工作量比 6:4，全年最大小时需求 = 0.85 × GPU 上限。"""
    az = azure_shapes(); B = ali_days()
    doy = np.array([d.timetuple().tm_yday for d in dates]); wd = np.array([d.weekday() for d in dates])
    season = 1 + 0.05 * np.cos(2 * np.pi * (doy - 196) / 365)
    R = np.where((wd < 5)[:, None], az["f_wd"] * az["wd"][None, :], az["f_we"] * az["we"][None, :]) * season[:, None]
    Bd = B[np.arange(len(dates)) % len(B)] * season[:, None]
    rR, rB = P["RB_RATIO"]
    R = R / R.sum() * rR; Bd = Bd / Bd.sum() * rB
    D = P["D_DEADLINE"]   # 可延期任务在截止期内可平移，峰值需求按其 D 小时滑动平均计
    Bm = np.array([np.convolve(np.r_[b, b], np.ones(D) / D)[24:48] for b in Bd])
    k = P["PEAK_RATIO"] * P["N_CAP"] / (R + Bm).max()
    return R * k, Bd * k


# ---------------------------------------------------------------- 典型日
def kmedoids(X, K, n_init=50, seed=0):
    """k-medoids（k-medoids++ 初始化 + 交替更新），多次初始化取总距离最小者。"""
    rng = np.random.default_rng(seed)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    best = None
    for _ in range(n_init):
        med = [rng.integers(len(X))]
        for _k in range(1, K):
            d2 = D[:, med].min(1) ** 2
            med.append(rng.choice(len(X), p=d2 / d2.sum()))
        med = np.array(med)
        for _it in range(100):
            lab = D[:, med].argmin(1)
            new = np.array([np.where(lab == k)[0][D[np.ix_(lab == k, lab == k)].sum(1).argmin()] for k in range(K)])
            if set(new) == set(med): break
            med = new
        lab = D[:, med].argmin(1); cost = D[np.arange(len(X)), med[lab]].sum()
        if best is None or cost < best[0]: best = (cost, med.copy(), lab.copy())
    _, med, lab = best
    order = np.argsort(med); med = med[order]; lab = np.argsort(order)[lab]
    return med, lab


def build(site, P=None, save=True):
    P = P or PR.params()
    lat, lon, _, yr = SITES[site]
    W = load_weather(site)
    dates = pd.date_range(f"{yr}-01-01", periods=365, freq="D")
    Tamb = W["temperature_2m"].values.reshape(365, 24)
    gPV = pv_output(W, lat, lon).reshape(365, 24)
    gWT, v = wind_output(W); gWT = gWT.reshape(365, 24)
    AR, AB = compute_load(P, dates)
    H, season = heat_demand(P, Tamb, dates)
    feats = dict(gPV=gPV, gWT=gWT, AR=AR, AB=AB, Tamb=Tamb, H=H)
    Z = np.concatenate([(a - a.min()) / (a.max() - a.min() + 1e-12) for a in feats.values()], axis=1)
    med, lab = kmedoids(Z, P["K"])
    p0 = np.bincount(lab, minlength=P["K"]) / 365
    th1, thi = PR.theta(P)
    out = dict(site=site, dates=np.array(dates.strftime("%Y-%m-%d"), dtype="U10"), heating=season, med=med, lab=lab, p0=p0,
               theta1=th1, thetainf=thi, wind_hub=v.reshape(365, 24), **feats)
    if save:
        np.savez(OUT / f"{site}_data.npz", **out)
    return out


def summary(D):
    s = D
    print(f"站点 {s['site']}: 年均气温 {s['Tamb'].mean():.1f} ℃, 最高 {s['Tamb'].max():.1f}, 最低 {s['Tamb'].min():.1f}; "
          f"光伏利用小时 {s['gPV'].sum():.0f} h, 风电利用小时 {s['gWT'].sum():.0f} h, 轮毂平均风速 {s['wind_hub'].mean():.2f} m/s")
    tot = s["AR"] + s["AB"]
    print(f"算力需求: 小时峰值 {tot.max():.0f} GPU, 均值 {tot.mean():.0f} GPU (平均利用率 {tot.mean() / 8000:.1%}), 实时:可延期 = "
          f"{s['AR'].sum() / tot.sum():.2f}:{s['AB'].sum() / tot.sum():.2f}")
    print(f"热负荷: 峰值 {s['H'].max():.0f} kW_th, 年总量 {s['H'].sum() / 1e3:.0f} MWh_th")
    print(f"典型日: 中心日 {[str(x) for x in s['dates'][s['med']]]}")
    print(f"p0 = {np.round(s['p0'], 3)}, θ1 = {s['theta1']:.4f}, θ∞ = {s['thetainf']:.4f}")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "ulanqab"
    summary(build(site))
