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


def semivar(Zs, mu, A):
    """各风险方向的正/负二阶半偏差 E[[a_ℓ^T(F−μ)]_+^2], E[[−a_ℓ^T(F−μ)]_+^2]。"""
    r = (Zs - mu) @ A.T
    return (np.clip(r, 0, None) ** 2).mean(0), (np.clip(-r, 0, None) ** 2).mean(0)


def bootstrap_bounds(Zs, mu, A, rng):
    """bootstrap: 均值误差 |mean_b − μ| 与半偏差的 BOOT_Q 分位数 (参考点均为 μ)。"""
    bm, bvp, bvn = [], [], []
    for _ in range(C.N_BOOT):
        Zb = Zs[rng.integers(0, len(Zs), len(Zs))]
        bm.append(np.abs(Zb.mean(0) - mu)); vp, vn = semivar(Zb, mu, A); bvp.append(vp); bvn.append(vn)
    q = lambda a: np.quantile(a, C.BOOT_Q, axis=0)
    return q(bm), q(bvp), q(bvn)


def calibrate(Z, A, nv, rng, n_split=50, n_boot=300):
    """验证集校准 (式17 "经验证集校准"): 重复随机划分估计集/验证集, 每次求使验证集均值偏差落入 δ 的最小放大系数 κ_δ,
    以及使验证集二阶半偏差落入 v± 的最小放大系数 κ_v; 取各划分的中位数 (≥1)。"""
    kd, kv = [], []
    for _ in range(n_split):
        idx = rng.permutation(len(Z)); val, est = idx[:nv], idx[nv:]
        mu_e = Z[est].mean(0)
        bm, bvp, bvn = [], [], []
        for _ in range(n_boot):
            Zb = Z[est][rng.integers(0, len(est), len(est))]
            bm.append(np.abs(Zb.mean(0) - mu_e)); vp, vn = semivar(Zb, mu_e, A); bvp.append(vp); bvn.append(vn)
        d0, vp0, vn0 = (np.quantile(x, C.BOOT_Q, axis=0) for x in (bm, bvp, bvn))
        vpe, vne = semivar(Z[est], mu_e, A); svp, svn = semivar(Z[val], mu_e, A)
        kd.append(np.max(np.abs(Z[val].mean(0) - mu_e) / np.maximum(d0, 1e-9)))
        kv.append(max(np.max((svp - vpe) / np.maximum(vp0 - vpe, 1e-9)), np.max((svn - vne) / np.maximum(vn0 - vne, 1e-9))))
    return float(max(1.0, np.median(kd))), float(max(1.0, np.median(kv)))


def nonempty_tau(F, mu, delta, rp, rn, vp, vn, gd, gp, gn):
    """LP: min τ ≥ 0, s.t. p ≥ 0, Σp = 1, |pF − μ| ≤ δ + τ gd, p rp ≤ vp + τ gp, p rn ≤ vn + τ gn。返回 τ (0 表示原集合非空)。"""
    from scipy.optimize import linprog
    S, nf = F.shape; nl = rp.shape[1]
    c = np.r_[np.zeros(S), 1.0]
    Aub = np.vstack([np.c_[F.T, -gd], np.c_[-F.T, -gd], np.c_[rp.T, -gp], np.c_[rn.T, -gn]])
    bub = np.r_[mu + delta, -(mu - delta), vp, vn]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=np.r_[np.ones(S), 0.0][None, :], b_eq=[1.0],
                bounds=[(0, None)] * S + [(0, None)], method="highs")
    assert r.status == 0, "模糊集非空性 LP 求解失败"
    return float(max(r.x[-1], 0.0)) * (1 + 1e-6) if r.x[-1] > 1e-9 else 0.0


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
    calib = []
    for name, (months, nd) in C.SEASONS.items():
        days = np.flatnonzero(np.isin(month, months)); rng.shuffle(days)
        d_tr, d_te = days[: int(0.67 * len(days))], days[int(0.67 * len(days)):]
        # 联合情景: 气象日 × 随机配对的负荷日(训练/测试负荷日互不重叠), 假设气象与算力需求相互独立
        J_tr = (d_tr, rng.choice(w_tr, len(d_tr))); J_te = (d_te, rng.choice(w_te, len(d_te)))
        F_tr = day_features(Tamb[J_tr[0]], gpv[J_tr[0]], AR[J_tr[1]], AB[J_tr[1]])
        mu_raw, sd_raw = F_tr.mean(0), F_tr.std(0) + 1e-9
        Z = (F_tr - mu_raw) / sd_raw                                    # 归一化特征
        med, p = kmedoids(Z, S, rng)
        L = [C.FEATURES.index(f) for f in C.RISK_DIRS.values()]
        sgn = np.array([+1, -1, +1])                                    # 高温(+), 低光伏(-), 高负荷(+)
        A = np.zeros((len(L), Z.shape[1])); A[range(len(L)), L] = sgn  # 风险方向 a_ℓ
        # ---- 式(17) 模糊集参数 ----
        # μ̂_d: 训练日条件经验特征均值;  ρ 与二阶半偏差全部以 μ̂_d 为参考点
        # δ^0, v^0: bootstrap 95% 分位数;  κ_cal: 估计集/验证集校准 (使验证集的均值偏差与半偏差落入集合的最小放大系数)
        for _ in range(C.N_BOOT): rng.integers(0, len(Z), len(Z))   # 保持主随机流与 v1 一致 → 训练/测试划分与情景不变
        rb = np.random.default_rng([C.SEED, len(sc["p"])])           # 校准与 bootstrap 使用独立随机流
        nv = int(round(C.VALID_FRAC * len(Z)))
        kd, kv = calibrate(Z, A, nv, rb)
        mu = Z.mean(0)
        d0, vp0, vn0 = bootstrap_bounds(Z, mu, A, rb)
        vpe, vne = semivar(Z, mu, A)
        delta = kappa * kd * d0
        v_p = vpe + kappa * kv * np.maximum(vp0 - vpe, 0.0); v_n = vne + kappa * kv * np.maximum(vn0 - vne, 0.0)
        if extreme:
            # 支持集扩充: 训练日中最高气温日、实时负荷峰值最大日、可延期负荷最大日 (经验概率 0, 由模糊集决定可分配概率)
            pk = AR[J_tr[1]].max(1)
            cand = [np.argsort(-F_tr[:, 1]), np.argsort(-pk), np.argsort(-F_tr[:, 4])]
            add = []
            for order in cand:
                add.append(int(next(i for i in order if i not in set(med) | set(add))))
            med = np.r_[med, add]; p = np.r_[p, np.zeros(len(add))]
        Fs = Z[med]
        rho = (Fs - mu) @ A.T
        rp, rn = np.clip(rho, 0, None) ** 2, np.clip(-rho, 0, None) ** 2
        # 非空性检验 (论文: 求解前验证 P_d ≠ ∅); 若为空, 求最小统一放大量 τ 并报告
        tau = nonempty_tau(Fs, mu, delta, rp, rn, v_p, v_n, d0, np.maximum(vp0 - vpe, 1e-6), np.maximum(vn0 - vne, 1e-6))
        if tau > 0:
            delta = delta + tau * d0
            v_p = v_p + tau * np.maximum(vp0 - vpe, 1e-6); v_n = v_n + tau * np.maximum(vn0 - vne, 1e-6)
        calib.append(dict(类别=name, kappa=kappa, kappa_delta=kd, kappa_v=kv, 非空放大tau=tau, 训练日=len(Z), 验证日=nv))
        for k, arr in zip(keys, [AR[J_tr[1][med]], AB[J_tr[1][med]], Tamb[J_tr[0][med]], gpv[J_tr[0][med]]]):
            sc[k].append(arr)
        for k, v in zip(["F", "p", "mu", "delta", "rho_p", "rho_n", "v_p", "v_n", "z_mu", "z_sd"],
                        [Fs, p, mu, delta, rp, rn, v_p, v_n, mu_raw, sd_raw]):
            sc[k].append(v)
        for k, arr in zip(keys, [AR[J_te[1]], AB[J_te[1]], Tamb[J_te[0]], gpv[J_te[0]]]):
            test[k].append(arr)
        test["F"].append((day_features(Tamb[J_te[0]], gpv[J_te[0]], AR[J_te[1]], AB[J_te[1]]) - mu_raw) / sd_raw)
    out["scen"] = {k: np.array(v) for k, v in sc.items()}
    out["calib"] = calib
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
    print("校准:", D["calib"])
    print("电价:", D["price"].round(4))
    print("年均气温 %.1f℃, 年光伏利用小时 %.0f h" % (D["Tamb_all"].mean(), D["gpv_all"].sum()))
