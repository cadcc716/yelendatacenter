"""
修订版实验方案 (E1–E5) 的数据集构建
  5.1.2 时间顺序划分: 每月按日期先后 前 60% 训练 / 15% 验证 / 25% 封存测试; 负荷轨迹同样按时间顺序分块
  联合日: 同一块内气象日与负荷日按日期顺序配对 (负荷轨迹仅 51 天, 按顺序循环), 保留各自日内时间结构
  有限支持: 每季节 k-medoids 代表日 + 训练集极端日 (最高气温日、实时负荷峰值日) 单独成簇 (概率 1/n_train)
  模糊集 (式30): U1 无条件对称 / U2 条件对称 / U3 条件正负半方差; δ、v 由训练集 bootstrap 得到,
                 再用验证块校准 (使验证块的均值偏差与二阶 (半) 矩落入集合的最小放大系数), 测试块不参与
  U0: 各季节训练日逐时均值轨迹 (确定性)
  b=0 静态冷却: 季节—时段的训练日平均气温 T̄_dt (预先冻结)
  连续压力片段: 测试块内 3 日连续片段 ST1 高温+低光伏 / ST2 高负荷+高温 / ST3 连续低光伏
运行: python data_e.py [--epw 文件名] [--tag 名称]
"""
import argparse, pickle
import numpy as np
import config as C
from data_prep import load_workload, load_weather, tou_price, semivar, nonempty_tau, kmedoids

FEAT = ["T_mean", "T_max", "PV_day", "R_peak", "B_day"]
RISK = {"高温": ("T_max", +1), "低光伏": ("PV_day", -1), "高负荷": ("B_day", +1)}


def feats(T, g, AR, AB):
    return np.c_[T.mean(1), T.max(1), g.sum(1), AR.max(1), AB.sum(1)]


def split_days(month, frac=C.SPLIT):
    tr, va, te = [], [], []
    for m in range(1, 13):
        idx = np.flatnonzero(month == m); n = len(idx)
        a, b = int(round(frac[0] * n)), int(round((frac[0] + frac[1]) * n))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def boot(Z, mu, A, rng, nb=C.N_BOOT):
    """bootstrap: 均值偏差、正负半方差、对称二阶矩 (均以 μ̂ 为参考点) 的 BOOT_Q 分位数。"""
    bm, bp, bn, bs = [], [], [], []
    for _ in range(nb):
        Zb = Z[rng.integers(0, len(Z), len(Z))]
        bm.append(np.abs(Zb.mean(0) - mu)); vp, vn = semivar(Zb, mu, A); bp.append(vp); bn.append(vn); bs.append(vp + vn)
    q = lambda a: np.quantile(a, C.BOOT_Q, axis=0)
    return q(bm), q(bp), q(bn), q(bs)


def amb_set(Ztr, Zva, Fs, A, kind, kappa, rng):
    """式(30) 有限支持模糊集参数. kind: 'sym' (U1/U2 对称二阶矩) 或 'semi' (U3 正负半方差)。"""
    mu = Ztr.mean(0)
    d0, vp0, vn0, vs0 = boot(Ztr, mu, A, rng)
    vpe, vne = semivar(Ztr, mu, A); vse = vpe + vne
    svp, svn = semivar(Zva, mu, A); svs = svp + svn
    kd = max(1.0, float(np.max(np.abs(Zva.mean(0) - mu) / np.maximum(d0, 1e-9))))
    if kind == "semi":
        kv = max(1.0, float(max(np.max((svp - vpe) / np.maximum(vp0 - vpe, 1e-9)), np.max((svn - vne) / np.maximum(vn0 - vne, 1e-9)))))
    else:
        kv = max(1.0, float(np.max((svs - vse) / np.maximum(vs0 - vse, 1e-9))))
    delta = kappa * kd * d0
    rho = (Fs - mu) @ A.T
    if kind == "semi":
        gp, gn = np.maximum(vp0 - vpe, 1e-6), np.maximum(vn0 - vne, 1e-6)
        v_p, v_n = vpe + kappa * kv * gp, vne + kappa * kv * gn
        rp, rn = np.clip(rho, 0, None) ** 2, np.clip(-rho, 0, None) ** 2
    else:
        gp = np.maximum(vs0 - vse, 1e-6); gn = np.full_like(gp, 1e-6)
        v_p, v_n = vse + kappa * kv * gp, np.zeros_like(gp)
        rp, rn = rho ** 2, np.zeros_like(rho)
    tau = nonempty_tau(Fs, mu, delta, rp, rn, v_p, v_n, d0, gp, gn)
    if tau > 0:
        delta = delta + tau * d0; v_p = v_p + tau * gp; v_n = v_n + tau * gn
    return dict(F=Fs, mu=mu, delta=delta, rho_p=rp, rho_n=rn, v_p=v_p, v_n=v_n,
                cal=dict(kappa=kappa, kappa_delta=kd, kappa_v=kv, tau=tau))


def ashrae_extreme(epw, n=C.DESIGN_RETURN_N):
    """EPW 文件头 DESIGN CONDITIONS (ASHRAE 2009 气候设计数据) 中 n 年一遇极端最高干球温度 (n = 5/10/20/50)。"""
    f = open(C.RAW / epw, encoding="latin-1").readlines()[1].strip().split(",")
    i = f.index("Extremes"); pos = {5: 10, 10: 12, 20: 14, 50: 16}[n]   # Extremes 段: n=5,10,20,50 年一遇 (min, max) 成对
    return float(f[i + pos])


def build(epw="USA_CO_Denver.Intl.AP.725650_TMY3.epw", r_mean=C.R_MEAN_GPU, b_mean=C.B_MEAN_GPU, kappa=1.0, train_frac=1.0):
    rng = np.random.default_rng(C.SEED)
    AR, AB = load_workload()
    AR, AB = AR / C.R_MEAN_GPU * r_mean, AB / C.B_MEAN_GPU * b_mean
    Tamb, gpv, month, meta = load_weather(epw)
    nW = len(AR); la, lb = int(round(C.SPLIT[0] * nW)), int(round(sum(C.SPLIT) * nW))
    Ltr, Lva, Lte = np.arange(la), np.arange(la, lb), np.arange(lb, nW)
    Wtr, Wva, Wte = split_days(month)
    if train_frac < 1.0:   # 训练样本量实验: 每月只保留训练块中最早的 train_frac 比例日 (验证/测试块不变)
        keep = []
        for m in range(1, 13):
            idx = [w for w in Wtr if month[w] == m]; keep += idx[:max(3, int(round(train_frac * len(idx))))]
        Wtr = np.array(keep)
    pair = lambda W, L: (W, L[np.arange(len(W)) % len(L)])           # 同块内按日期顺序配对
    blocks = {"tr": pair(Wtr, Ltr), "va": pair(Wva, Lva), "te": pair(Wte, Lte)}
    season = {m: i for i, (_, (ms, _)) in enumerate(C.SEASONS.items()) for m in ms}
    out = dict(price=tou_price(), classes=list(C.SEASONS), n_d=np.array([v[1] for v in C.SEASONS.values()]),
               site=meta, epw=epw, AR_all=AR, AB_all=AB, Tamb_all=Tamb, gpv_all=gpv, month=month, split=blocks)
    A = np.zeros((len(RISK), len(FEAT)))
    for i, (f, sg) in enumerate(RISK.values()): A[i, FEAT.index(f)] = sg
    out["risk_A"] = A
    day = lambda w, l: dict(AR=AR[l], AB=AB[l], Tamb=Tamb[w], gpv=gpv[w], w=int(w), l=int(l), month=int(month[w]))
    # ---------- 季节条件支持集 ----------
    T_des = ashrae_extreme(epw)
    hot_w = max(blocks["tr"][0], key=lambda w: Tamb[w].max()); hot_d = season[month[hot_w]]
    out["design"] = dict(T=T_des, base_day=int(hot_w), season=int(hot_d), n=C.DESIGN_RETURN_N)
    sup, ambs, cal, U0, Tbar, val_days, test_days = [], {"U2": [], "U3": []}, [], [], [], [], []
    Fall_tr, Fall_va, sup_raw = [], [], []
    for d, name in enumerate(C.SEASONS):
        sel = lambda key: [(w, l) for w, l in zip(*blocks[key]) if season[month[w]] == d]
        tr, va, te = sel("tr"), sel("va"), sel("te")
        Ftr = feats(Tamb[[w for w, _ in tr]], gpv[[w for w, _ in tr]], AR[[l for _, l in tr]], AB[[l for _, l in tr]])
        Fva = feats(Tamb[[w for w, _ in va]], gpv[[w for w, _ in va]], AR[[l for _, l in va]], AB[[l for _, l in va]])
        m0, s0 = Ftr.mean(0), Ftr.std(0) + 1e-9
        Ztr, Zva = (Ftr - m0) / s0, (Fva - m0) / s0
        ext = []
        for f in C.EXTREMES:
            order = np.argsort(-Ftr[:, FEAT.index(f)])
            ext.append(int(next(i for i in order if i not in ext)))
        rest = np.setdiff1d(np.arange(len(tr)), ext)
        med, pr = kmedoids(Ztr[rest], C.N_MED, rng)
        idx = list(rest[med]) + ext
        p = np.r_[pr * len(rest) / len(tr), np.full(len(ext), 1.0 / len(tr))]
        tag = [f"代表日{i + 1}" for i in range(C.N_MED)] + [f"极端:{f}" for f in C.EXTREMES]
        Fs = Ztr[idx]
        dsg = None
        if d == hot_d:
            # 设计极端日 (5.4.2 独立历史资料): 训练集最热日的逐时温度整体平移至 ASHRAE n 年一遇极端最高温度,
            # 负荷与光伏取该日实况; 经验概率 0, 只作为支撑点 (运行可行性硬约束 + 模糊集可分配概率)
            j = [k for k, (w, _) in enumerate(tr) if w == hot_w][0]; w0 = tr[j][0]
            l0 = int(max(Ltr, key=lambda l: AR[l].sum() + AB[l].sum()))   # 设计负荷日: 训练块日负荷最大的一天 (各气候相同)
            Td = Tamb[w0] + (T_des - Tamb[w0].max())
            Fd = feats(Td[None], gpv[[w0]], AR[[l0]], AB[[l0]]); Fs = np.vstack([Fs, (Fd - m0) / s0])
            p = np.r_[p, 0.0]; tag = tag + [f"设计极端日{T_des:.1f}℃"]
            dsg = dict(d=d, s=len(idx), p=0.0, tag=tag[-1], AR=AR[l0], AB=AB[l0], Tamb=Td, gpv=gpv[w0], w=int(w0), l=int(l0),
                       month=int(month[w0]))
        for kind, key in (("sym", "U2"), ("semi", "U3")):
            a = amb_set(Ztr, Zva, Fs, A, kind, kappa, np.random.default_rng([C.SEED, d, len(key)]))
            ambs[key].append(a); cal.append(dict(模型=key, 类别=name, 训练日=len(tr), 验证日=len(va), **a["cal"]))
        sup.append([dict(d=d, s=j, p=float(p[j]), tag=tag[j], **day(*tr[i])) for j, i in enumerate(idx)] + ([dsg] if dsg else []))
        Fall_tr.append(Ftr); Fall_va.append(Fva); sup_raw.append(Fs * s0 + m0)
        Tm = Tamb[[w for w, _ in tr]].mean(0); Tbar.append(Tm)
        U0.append(dict(d=d, s=0, p=1.0, tag="均值轨迹", AR=AR[[l for _, l in tr]].mean(0), AB=AB[[l for _, l in tr]].mean(0),
                       Tamb=Tm, gpv=gpv[[w for w, _ in tr]].mean(0), w=-1, l=-1, month=0))
        val_days.append([dict(d=d, **day(w, l)) for w, l in va]); test_days.append([dict(d=d, **day(w, l)) for w, l in te])
    # ---------- U1: 无条件 (全年合并支持集与统计量) ----------
    Ftr_all, Fva_all, Fs_all = np.vstack(Fall_tr), np.vstack(Fall_va), np.vstack(sup_raw)
    m0, s0 = Ftr_all.mean(0), Ftr_all.std(0) + 1e-9
    a1 = amb_set((Ftr_all - m0) / s0, (Fva_all - m0) / s0, (Fs_all - m0) / s0, A, "sym", kappa, np.random.default_rng([C.SEED, 99]))
    cal.append(dict(模型="U1", 类别="全年", 训练日=len(Ftr_all), 验证日=len(Fva_all), **a1["cal"]))
    out.update(sup=sup, amb=ambs, amb_U1=a1, calib=cal, U0=U0, Tbar=np.array(Tbar), val=val_days, test=test_days)
    out["stress"] = stress_segments(out)
    return out


def stress_segments(D, L=3):
    """测试块内的连续 L 日片段 (同月连续日期), 按训练集统计预先定义的三类压力得分选取, 不使用模型结果。"""
    Tm, g, AR, AB = D["Tamb_all"], D["gpv_all"], D["AR_all"], D["AB_all"]
    W, Lw = D["split"]["te"]
    cand = []
    for i in range(len(W) - L + 1):
        w, l = W[i:i + L], Lw[i:i + L]
        if not (np.all(np.diff(w) == 1) and len(set(D["month"][w])) == 1): continue
        cand.append((w, l))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    Tmax = np.array([Tm[w].max(1).mean() for w, _ in cand]); PV = np.array([g[w].sum() for w, _ in cand])
    Rpk = np.array([AR[l].max() for _, l in cand]); Bd = np.array([AB[l].sum() for _, l in cand])
    hot = Tmax >= np.quantile(Tmax, 0.75)
    score = {"ST1 高温+低光伏": np.where(PV <= np.median(PV), Tmax, -np.inf),        # 光伏低于中位数的片段中最热者
             "ST2 高负荷+高温": np.where(hot, z(Rpk) + z(Bd), -np.inf),               # 高温四分位片段中负荷最高者
             "ST3 连续低光伏": -PV}                                                   # 3 日光伏电量最低
    seg = {}
    for k, sc in score.items():
        w, l = cand[int(np.argmax(sc))]
        seg[k] = dict(w=w, l=l, AR=AR[l].ravel(), AB=AB[l].ravel(), Tamb=Tm[w].ravel(), gpv=g[w].ravel(),
                      d=int({m: i for i, (_, (ms, _)) in enumerate(C.SEASONS.items()) for m in ms}[int(D["month"][w[0]])]))
    return seg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw", default="USA_CO_Denver.Intl.AP.725650_TMY3.epw"); ap.add_argument("--tag", default="base")
    ap.add_argument("--r_mean", type=float, default=C.R_MEAN_GPU); ap.add_argument("--b_mean", type=float, default=C.B_MEAN_GPU)
    ap.add_argument("--kappa", type=float, default=1.0)
    a = ap.parse_args()
    D = build(a.epw, a.r_mean, a.b_mean, a.kappa)
    pickle.dump(D, open(C.PROC / f"dataset_{a.tag}.pkl", "wb"))
    import pandas as pd
    print(pd.DataFrame(D["calib"]).round(3).to_string())
    for d, S in enumerate(D["sup"]):
        print(D["classes"][d], [(b["tag"], round(b["p"], 3), round(b["Tamb"].max(), 1), round(b["AR"].max())) for b in S])
    print("测试日数:", [len(t) for t in D["test"]], "验证日数:", [len(t) for t in D["val"]])
    for k, s in D["stress"].items():
        print(k, "月", D["month"][s["w"]], "日", s["w"], "Tmax", round(s["Tamb"].max(), 1), "PV", round(s["gpv"].sum(), 2),
              "R峰", round(s["AR"].max()))
