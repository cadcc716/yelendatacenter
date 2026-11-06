"""
数据准备（执行方案 5.1 节、式(30)(31)）
  气象：中卫为 Open-Meteo（ERA5）逐时 JSON，贵安、韶关为 NASA POWER 逐时 JSON（2023 年）；
        三地统一用水平总辐照 GHI → Erbs 直散分离 → 倾斜面辐照，换算为光伏单位出力系数。
        中卫以光伏资源为主、风资源不足，不配置风电
  算力负荷：Azure LLM 推理 trace（2024-05 一周）→ 实时任务日内形状与工作日/周末系数
            Alibaba cluster-trace-gpu-v2020（PAI 任务表）→ 可延期任务逐日到达形状
  热负荷：采暖季（11-01 至次年 03-31）随日均气温线性变化，峰值 3 MW_th；非采暖季 0.3 MW_th 生活热水；
          贵安、韶关无集中供热，仅保留 0.3 MW_th 生活热水
  典型日：5 类日轨迹（光伏、实时任务、可延期任务、气温、热负荷）联合归一化，k-medoids 聚为 K 个典型日，输出 ξ_s、p0、θ1、θ∞

用法：python data_prep.py <site>      site ∈ SITES
原始 trace 的逐时汇总（v3/data/trace/*.csv）由 aggregate_traces() 从原始文件生成。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import params as PR

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "v3"; OUT = V3 / "data"; TRACE = OUT / "trace"
SITES = {   # 名称: (纬度, 经度, 气象文件, 年份, 是否集中供热)；基准算例为宁夏中卫（东数西算宁夏枢纽中卫集群）
    "zhongwei": (37.50, 105.22, ROOT / "data/raw/zhongwei_era5.json", 2023, True),
    "guian": (26.40, 106.38, ROOT / "data/raw/guian_2023.json", 2023, False),
    "shaoguan": (24.81, 113.60, ROOT / "data/raw/shaoguan_2023.json", 2023, False),
}


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
    """返回北京时间逐时序列（8760 行），列 Tamb、GHI；时刻 t 的辐照为 [t, t+1) 时段平均。
    Open-Meteo：辐照为前一小时平均、标注在时段末，前移 1 h；气温为瞬时值。
    NASA POWER：LST = UTC + round(经度/15)，辐照为 [t, t+1) 平均，换算到北京时间（UTC+8）。"""
    lat, lon, f, yr, _ = SITES[site]
    if not f.exists(): raise FileNotFoundError(f"缺少气象文件 {f}")
    d = json.load(open(f)); idx = pd.date_range(f"{yr}-01-01", f"{yr}-12-31 23:00", freq="h")
    if "hourly" in d:                                       # Open-Meteo
        W = pd.DataFrame(d["hourly"]); W["time"] = pd.to_datetime(W["time"]); W = W.set_index("time")
        W = W[~W.index.duplicated()]
        if d.get("utc_offset_seconds", 0) != 28800:
            W.index = W.index + pd.Timedelta(seconds=28800 - d.get("utc_offset_seconds", 0))
        T = W["temperature_2m"].reindex(idx)
        G = W["shortwave_radiation"].shift(-1, freq="h").reindex(idx)
    else:                                                   # NASA POWER
        par = d["properties"]["parameter"]; off = 8 - round(lon / 15)
        t = pd.to_datetime(list(par["T2M"].keys()), format="%Y%m%d%H") + pd.Timedelta(hours=off)
        T = pd.Series(list(par["T2M"].values()), index=t, dtype=float).reindex(idx)
        G = pd.Series(list(par["ALLSKY_SFC_SW_DWN"].values()), index=t, dtype=float).reindex(idx)
        T[T <= -999] = np.nan; G[G <= -999] = np.nan
    W = pd.DataFrame({"Tamb": T, "GHI": G}).interpolate(limit_direction="both")
    W["GHI"] = W["GHI"].clip(lower=0)
    return W


def pv_output(W, lat, lon):
    """固定倾角（=纬度、正南）组件：GHI 经 Erbs 模型分离直射与散射，Hay-Davies 模型换算倾斜面辐照；
    电池温度按 NOCT 法，温度系数 −0.37 %/K，系统综合效率 0.86。太阳位置取时段中点。返回单位装机出力系数。"""
    import pvlib
    t = W.index.tz_localize("Asia/Shanghai")
    sp = pvlib.solarposition.get_solarposition(t + pd.Timedelta("30min"), lat, lon)
    zen = sp["apparent_zenith"].values; azi = sp["azimuth"].values; ghi = W["GHI"].values
    dec = pvlib.irradiance.erbs(ghi, zen, t + pd.Timedelta("30min"))
    dni = np.nan_to_num(np.asarray(dec["dni"], float)); dhi = np.nan_to_num(np.asarray(dec["dhi"], float))
    dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(t), float)
    poa = pvlib.irradiance.get_total_irradiance(abs(lat), 180, zen, azi, dni, ghi, dhi, dni_extra=dni_extra, model="haydavies")
    G = np.nan_to_num(np.asarray(poa["poa_global"], float)).clip(0)
    Tc = W["Tamb"].values + (45 - 20) / 800 * G
    return np.clip(G / 1000 * (1 - 0.0037 * (Tc - 25)) * 0.86, 0, 1)


# ---------------------------------------------------------------- 负荷与热负荷
def heat_demand(P, Tamb_day, dates, district=True):
    """采暖季热负荷 ∝ (18 − 日均气温)，叠加生活热水；日内形状取居民/办公典型曲线。"""
    heat_shape = np.array([1.10, 1.12, 1.14, 1.15, 1.15, 1.18, 1.20, 1.15, 1.02, 0.92, 0.85, 0.80,
                           0.78, 0.78, 0.80, 0.84, 0.90, 0.98, 1.05, 1.08, 1.08, 1.06, 1.05, 1.08])
    dhw_shape = np.array([0.35, 0.30, 0.30, 0.30, 0.40, 0.70, 1.30, 1.60, 1.30, 1.00, 0.90, 0.90,
                          1.00, 0.90, 0.80, 0.80, 0.90, 1.10, 1.40, 1.70, 1.80, 1.50, 1.00, 0.60])
    heat_shape = heat_shape / heat_shape.max(); dhw_shape = dhw_shape / dhw_shape.max()
    md = np.array([d.month * 100 + d.day for d in dates])
    season = ((md >= 1101) | (md <= 331)) & district
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
    lat, lon, _, yr, district = SITES[site]
    W = load_weather(site)
    dates = pd.date_range(f"{yr}-01-01", periods=365, freq="D")
    Tamb = W["Tamb"].values.reshape(365, 24)
    gPV = pv_output(W, lat, lon).reshape(365, 24)
    AR, AB = compute_load(P, dates)
    if district:
        H, season = heat_demand(P, Tamb, dates)
    else:                                                   # 无集中供热：仅生活热水
        H, season = heat_demand(P, Tamb, dates, district=False)
    feats = dict(gPV=gPV, AR=AR, AB=AB, Tamb=Tamb, H=H)
    Z = np.concatenate([(a - a.min()) / (a.max() - a.min() + 1e-12) for a in feats.values()], axis=1)
    med, lab = kmedoids(Z, P["K"])
    p0 = np.bincount(lab, minlength=P["K"]) / 365
    th1, thi = PR.theta(P)
    out = dict(site=site, dates=np.array(dates.strftime("%Y-%m-%d"), dtype="U10"), heating=season, med=med, lab=lab, p0=p0,
               theta1=th1, thetainf=thi, **feats)
    if save:
        np.savez(OUT / f"{site}_data.npz", **out)
    return out


def summary(D):
    s = D
    print(f"站点 {s['site']}: 年均气温 {s['Tamb'].mean():.1f} ℃, 最高 {s['Tamb'].max():.1f}, 最低 {s['Tamb'].min():.1f}; "
          f"光伏利用小时 {s['gPV'].sum():.0f} h")
    tot = s["AR"] + s["AB"]
    print(f"算力需求: 小时峰值 {tot.max():.0f} GPU, 均值 {tot.mean():.0f} GPU (平均利用率 {tot.mean() / 8000:.1%}), 实时:可延期 = "
          f"{s['AR'].sum() / tot.sum():.2f}:{s['AB'].sum() / tot.sum():.2f}")
    print(f"热负荷: 峰值 {s['H'].max():.0f} kW_th, 年总量 {s['H'].sum() / 1e3:.0f} MWh_th")
    print(f"典型日: 中心日 {[str(x) for x in s['dates'][s['med']]]}")
    print(f"p0 = {np.round(s['p0'], 3)}, θ1 = {s['theta1']:.4f}, θ∞ = {s['thetainf']:.4f}")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    summary(build(site))


def add_extremes(site, n_max=4):
    """极端日试验：在 12 个典型日之外加入极端日作为小概率场景（p0 = 1/365），所在簇概率相应扣减。
    依次取：全年算力峰值小时所在日、日算力需求最大日、最高气温日、日均气温最高日（重复则跳过）。"""
    D = dict(np.load(OUT / f"{site}_data.npz")); P = PR.params()
    AR, AB, T = D["AR"], D["AB"], D["Tamb"]; Dd = P["D_DEADLINE"]
    Bm = np.array([np.convolve(np.r_[b, b], np.ones(Dd) / Dd)[24:48] for b in AB])
    cand = [("算力峰值日", int(np.argmax((AR + Bm).max(1)))), ("日需求最大日", int(np.argmax((AR + AB).sum(1)))),
            ("最高气温日", int(np.argmax(T.max(1)))), ("日均气温最高日", int(np.argmax(T.mean(1))))]
    med = list(D["med"]); lab = D["lab"]; cnt = np.bincount(lab, minlength=len(med)).astype(float); ext = []
    for name, d in cand:
        if d in med or d in [e[1] for e in ext]: continue
        ext.append((name, d)); cnt[lab[d]] -= 1
        if len(ext) >= n_max: break
    p0 = np.r_[cnt, np.ones(len(ext))] / 365
    K = len(med) + len(ext); th1, thi = PR.theta(P, K=K)
    out = dict(D); out.update(med=np.array(med + [e[1] for e in ext]), p0=p0, theta1=th1, thetainf=thi,
                              ext_names=np.array([e[0] for e in ext], dtype="U10"), ext_days=np.array([e[1] for e in ext]))
    np.savez(OUT / f"{site}_ext_data.npz", **out)
    for name, d in ext:
        print(f"  {name}: {D['dates'][d]}，最高气温 {T[d].max():.1f} ℃，小时峰值需求 {(AR + Bm)[d].max():.0f} GPU，日需求 {(AR + AB)[d].sum():.0f} GPU·h")
    print(f"K = {K}，θ1 = {th1:.4f}，θ∞ = {thi:.4f}；典型日最大小时需求 {(AR + Bm)[D['med']].max():.0f} GPU")
    return out
