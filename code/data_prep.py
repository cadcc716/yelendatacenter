"""
数据读取与情景生成
  1) BurstGPT 真实 LLM 服务轨迹 -> 实时(对话)/可延期(API)工作量 A^R, A^B
  2) EPW 典型气象年 -> 环境温度 T_amb 与光伏单位出力 g^PV (pvlib, PVWatts v5)
  3) 宁夏峰谷分时电价 c^e_t
  4) 季节代表类别 d, 联合日情景池 -> 训练/测试划分 -> k-medoids 有限支持情景 s 与经验概率
  5) 式(17) 模糊集参数: 特征 F_ds, μ̂_d, δ_d, 风险方向偏差 ρ_ℓ,ds, v±_dℓ (bootstrap 估计)
运行: python code/data_prep.py [--epw 文件名]  (替换为中卫 CSWD EPW 即可切换真实站点)
"""
import argparse, pickle, urllib.request
import numpy as np, pandas as pd, pvlib
import config as C

BURST_URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"


def load_workload():
    """BurstGPT_1 (Wang et al., KDD'25): Azure 上 ChatGPT/GPT-4 服务 61 天请求级轨迹(本地时间)。
    对话日志 -> 实时任务 R; API 日志 -> 可延期任务 B。返回逐时 token 量 (天 × 24)。"""
    hourly = C.PROC / "burstgpt_hourly.csv"
    if not hourly.exists():
        raw = C.RAW / "BurstGPT_1.csv"
        if not raw.exists():
            print("下载 BurstGPT_1.csv ..."); urllib.request.urlretrieve(BURST_URL, raw)
        df = pd.read_csv(raw, usecols=["Timestamp", "Total tokens", "Log Type"])
        df["h"] = (df.Timestamp // 3600).astype(int)
        g = df.groupby(["h", "Log Type"])["Total tokens"].sum().unstack(fill_value=0)
        g = g.reindex(range(61 * 24), fill_value=0)
        g.columns = ["API_tokens" if "API" in c else "Conv_tokens" for c in g.columns]
        g.index.name = "hour"; g.to_csv(hourly)
    g = pd.read_csv(hourly, index_col=0)
    R = shift_days(g["Conv_tokens"].values)[C.BURST_DAY0:]
    B = shift_days(g["API_tokens"].values)[C.BURST_DAY0:]
    # 规模化: 令年均需求分别为 R_MEAN_GPU, B_MEAN_GPU (GPU·h_nom/h), 保留真实日内形状与日间波动
    return R / R.mean() * C.R_MEAN_GPU, B / B.mean() * C.B_MEAN_GPU


def shift_days(x):
    """逐时序列 -> (日 × 24), 每个调度日从当地 H0 时起算 (最后不完整的一天舍去)。"""
    x = np.asarray(x, float)[C.H0:]
    n = len(x) // 24
    return x[: n * 24].reshape(n, 24)


def load_weather(epw):
    """EPW 逐时干球温度与辐照 -> PVWatts v5 光伏单位出力 (kW_ac / kWp_dc)。"""
    data, meta = pvlib.iotools.read_epw(C.RAW / epw)
    data = data.iloc[:8760]
    loc = pvlib.location.Location(meta["latitude"], meta["longitude"], tz=f"Etc/GMT{-int(meta['TZ']):+d}",
                                  altitude=meta["altitude"])
    t_mid = data.index - pd.Timedelta("30min")                          # EPW 为时段末标记
    sp = loc.get_solarposition(t_mid)
    poa = pvlib.irradiance.get_total_irradiance(C.PV_TILT, C.PV_AZ, sp.apparent_zenith.values, sp.azimuth.values,
                                                data.dni.values, data.ghi.values, data.dhi.values,
                                                dni_extra=pvlib.irradiance.get_extra_radiation(t_mid).values,
                                                model="haydavies")
    prm = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    tc = pvlib.temperature.sapm_cell(poa["poa_global"], data.temp_air.values, data.wind_speed.values, **prm)
    pdc = pvlib.pvsystem.pvwatts_dc(poa["poa_global"], tc, 1.0, C.PV_GAMMA) * (1 - C.PV_LOSS)
    pac0 = 1 / 1.2                                                      # DC/AC 容配比 1.2
    pac = pvlib.inverter.pvwatts(np.nan_to_num(np.asarray(pdc, float)), pac0 / C.PV_INV_EFF, C.PV_INV_EFF)
    pac = np.clip(np.nan_to_num(np.asarray(pac, float)), 0, pac0)
    month = shift_days(data.index.month.values)[:, 0].astype(int)
    return shift_days(data.temp_air.values), shift_days(pac), month, meta


def tou_price():
    """宁夏工商业峰谷分时电价: 交易电价按平段±FLOAT 浮动, 输配电价不随时段浮动。"""
    k = np.ones(24); k[C.PEAK_H] += C.FLOAT; k[C.VALLEY_H] -= C.FLOAT
    return np.roll(C.LAMBDA_E * k + C.C_TD, -C.H0)          # 与调度日窗口对齐 (t=1 ↔ H0 时)


def day_features(Tamb, gpv, AR, AB):
    return np.c_[Tamb.mean(1), Tamb.max(1), gpv.sum(1), AR.sum(1), AB.sum(1)]


def kmedoids(Z, k, rng, it=100):
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = [int(np.argmin(D.sum(1)))]                                  # k-means++ 式初始化
    while len(med) < k:
        dmin = D[:, med].min(1); med.append(int(rng.choice(len(Z), p=dmin / dmin.sum())))
    for _ in range(it):
        lab = D[:, med].argmin(1)
        new = [int(np.flatnonzero(lab == c)[np.argmin(D[np.ix_(lab == c, lab == c)].sum(1))]) for c in range(k)]
        if new == med: break
        med = new
    lab = D[:, med].argmin(1)
    return np.array(med), np.bincount(lab, minlength=k) / len(Z)


def build(epw="USA_CO_Denver.Intl.AP.725650_TMY3.epw", kappa=1.0, extreme=False):
    rng = np.random.default_rng(C.SEED)
    AR, AB = load_workload()
    Tamb, gpv, month, meta = load_weather(epw)
    price = tou_price()
    nW = len(AR); perm = rng.permutation(nW); w_tr, w_te = perm[: int(0.67 * nW)], perm[int(0.67 * nW):]
    out = dict(price=price, classes=list(C.SEASONS), n_d=np.array([v[1] for v in C.SEASONS.values()]),
               site=meta, epw=epw, AR_all=AR, AB_all=AB, Tamb_all=Tamb, gpv_all=gpv, month=month)
    S, keys = C.N_SCEN, ["AR", "AB", "Tamb", "gpv"]
    sc = {k: [] for k in keys + ["F", "p", "mu", "delta", "rho_p", "rho_n", "v_p", "v_n", "z_mu", "z_sd"]}
    test = {k: [] for k in keys + ["F"]}
    for name, (months, nd) in C.SEASONS.items():
        days = np.flatnonzero(np.isin(month, months)); rng.shuffle(days)
        d_tr, d_te = days[: int(0.67 * len(days))], days[int(0.67 * len(days)):]
        # 联合情景: 气象日 × 随机配对的负荷日(训练/测试负荷日互不重叠), 假设气象与算力需求相互独立
        J_tr = (d_tr, rng.choice(w_tr, len(d_tr))); J_te = (d_te, rng.choice(w_te, len(d_te)))
        F_tr = day_features(Tamb[J_tr[0]], gpv[J_tr[0]], AR[J_tr[1]], AB[J_tr[1]])
        mu_raw, sd_raw = F_tr.mean(0), F_tr.std(0) + 1e-9
        Z = (F_tr - mu_raw) / sd_raw                                    # 归一化特征
        med, p = kmedoids(Z, S, rng)
        Fs = Z[med]
        mu = p @ Fs                                                     # 有限支持上的条件经验特征均值 μ̂_d
        # bootstrap: 均值误差界 δ_d 与各风险方向二阶半偏差界 v±
        L = [C.FEATURES.index(f) for f in C.RISK_DIRS.values()]
        sgn = np.array([+1, -1, +1])                                    # 高温(+), 低光伏(-), 高负荷(+)
        A = np.zeros((len(L), Z.shape[1])); A[range(len(L)), L] = sgn  # 风险方向 a_ℓ
        bm, bvp, bvn = [], [], []
        for _ in range(C.N_BOOT):
            Zb = Z[rng.integers(0, len(Z), len(Z))]
            mb = Zb.mean(0); bm.append(np.abs(mb - Z.mean(0)))
            r = (Zb - Z.mean(0)) @ A.T
            bvp.append((np.clip(r, 0, None) ** 2).mean(0)); bvn.append((np.clip(-r, 0, None) ** 2).mean(0))
        delta = kappa * np.quantile(bm, C.BOOT_Q, axis=0)
        if extreme:
            # 支持集扩充: 训练日中最高气温日、实时负荷峰值最大日、可延期负荷最大日 (经验概率 0, 由模糊集决定可分配概率)
            pk = AR[J_tr[1]].max(1)
            cand = [np.argsort(-F_tr[:, 1]), np.argsort(-pk), np.argsort(-F_tr[:, 4])]
            add = []
            for order in cand:
                add.append(int(next(i for i in order if i not in set(med) | set(add))))
            med = np.r_[med, add]; p = np.r_[p, np.zeros(len(add))]; Fs = Z[med]
        rho = (Fs - mu) @ A.T
        rp, rn = np.clip(rho, 0, None) ** 2, np.clip(-rho, 0, None) ** 2
        # 保证 p̂ ∈ P_d (模糊集非空): v 取 bootstrap 分位数与经验值的较大者, 再按 kappa 放缩
        v_p = np.maximum(p @ rp, np.quantile(bvp, C.BOOT_Q, axis=0)) * (1 + kappa) / 2 if kappa > 0 else p @ rp
        v_n = np.maximum(p @ rn, np.quantile(bvn, C.BOOT_Q, axis=0)) * (1 + kappa) / 2 if kappa > 0 else p @ rn
        for k, arr in zip(keys, [AR[J_tr[1][med]], AB[J_tr[1][med]], Tamb[J_tr[0][med]], gpv[J_tr[0][med]]]):
            sc[k].append(arr)
        for k, v in zip(["F", "p", "mu", "delta", "rho_p", "rho_n", "v_p", "v_n", "z_mu", "z_sd"],
                        [Fs, p, mu, delta, rp, rn, v_p, v_n, mu_raw, sd_raw]):
            sc[k].append(v)
        for k, arr in zip(keys, [AR[J_te[1]], AB[J_te[1]], Tamb[J_te[0]], gpv[J_te[0]]]):
            test[k].append(arr)
        test["F"].append((day_features(Tamb[J_te[0]], gpv[J_te[0]], AR[J_te[1]], AB[J_te[1]]) - mu_raw) / sd_raw)
        assert np.all(np.abs(p @ Fs - mu) <= delta + 1e-9), "模糊集为空"
    out["scen"] = {k: np.array(v) for k, v in sc.items()}
    out["test"] = test
    out["risk_A"] = A
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epw", default="USA_CO_Denver.Intl.AP.725650_TMY3.epw")
    a = ap.parse_args()
    with open(C.PROC / "scenarios_ext.pkl", "wb") as f: pickle.dump(build(a.epw, extreme=True), f)
    D = build(a.epw)
    with open(C.PROC / "scenarios.pkl", "wb") as f: pickle.dump(D, f)
    s = D["scen"]
    print("站点:", D["site"]["city"], D["site"]["latitude"], D["site"]["altitude"])
    print("代表类别:", D["classes"], "n_d =", D["n_d"])
    print("情景概率 p̂:\n", s["p"].round(3)); print("δ_d:\n", s["delta"].round(3))
    print("电价:", D["price"].round(4))
    print("年均气温 %.1f℃, 年光伏利用小时 %.0f h" % (D["Tamb_all"].mean(), D["gpv_all"].sum()))
