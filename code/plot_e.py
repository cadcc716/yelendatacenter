"""
修订版实验 E1–E5 图件 (对标 Applied Energy 等期刊参考文献的多种呈现形式)
规范: 中文宋体 + 英文/数字 Times New Roman; 坐标标题 10 pt, 刻度/图例 9 pt; 主线宽 1.1 pt; Marker 4.8 pt;
      白底; 图例无边框; 单栏 8.5 cm / 双栏 17.6 cm; 600 dpi PNG。
原则: 不改动数据; 柱状图从 0 起; 差值图保留零线; 误差线/标注给出求解间隙与失约, 避免把不可行方案的低成本当作优势。
运行: python plot_e.py [图号 ...]
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MPath
import config as C, model as Mo
from plot_style import style_b, CM

RES = C.RES / "exp"; OUT = C.FIG / "exp"; OUT.mkdir(parents=True, exist_ok=True)
style_b()
W1, W2 = 8.5 * CM, 17.6 * CM
COLM = {"M0": "#8C8C8C", "M1": "#4DBBD5", "M2": "#00A087", "M3": "#3C5488", "M0-W45": "#B09C85",
        "U0": "#F39B7F", "U1": "#91D1C2", "U2": "#8491B4", "U3": "#E64B35"}
LABM = {"M0": "M0 静态基准", "M1": "M1 +GPU调节", "M2": "M2 +热反馈", "M3": "M3 完整耦合", "M0-W45": "M0-W45 静态温水",
        "U0": "U0 确定性", "U1": "U1 无条件对称", "U2": "U2 条件对称", "U3": "U3 本文"}
DEV = {"PV": ("光伏", "#F6C85F"), "EB": ("储能", "#5AB4AC"), "FC": ("自然冷却", "#7FC97F"), "MC": ("机械制冷", "#F4A6A6"),
       "N": ("GPU", "#8FA9D6"), "PB": ("储能变流", "#9DD9D2")}
WAN = 1e4


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print("saved", name)


def S():
    return pd.read_csv(RES / "summary.csv", index_col=0)


def panel(ax, s, x=-0.16, y=1.02):
    (ax.text2D if hasattr(ax, "text2D") else ax.text)(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def legend_top(fig, handles, labels, ncol, y=1.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.4, frameon=False)


# ====================== 图2 GPU 运行模式: 处理速率—功率—单位能耗—热安全上限 ======================
def fig2():
    P = C.params(); ks = list(P["MODES"]); lab = ["k1 标称", "k2 浅锁频", "k3 深锁频"]
    v = np.array([P["MODES"][k][0] for k in ks]); pg = np.array([P["MODES"][k][1] for k in ks])
    ptot = pg + P["P_HOST"]; e = ptot / v
    tsafe = P["T_CHIP_MAX"] - P["R_TH"] * pg
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 5.6 * CM), layout="constrained", gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.arange(3); w = 0.34
    a.bar(x - w / 2, v * 100, w, color="#8FA9D6", edgecolor="#3C5488", lw=0.6, label="相对处理速率 $v_{rk}$")
    a.bar(x + w / 2, ptot / ptot[0] * 100, w, color="#F4A6A6", edgecolor="#C0392B", lw=0.6, label="整机功率 $p_{rk}$")
    for i in range(3):
        a.text(x[i] - w / 2, v[i] * 100 + 1, f"{v[i] * 100:.0f}", ha="center", fontsize=8)
        a.text(x[i] + w / 2, ptot[i] / ptot[0] * 100 + 1, f"{ptot[i] / ptot[0] * 100:.1f}", ha="center", fontsize=8)
    a.set_ylim(70, 108); a.set_ylabel("相对标称模式/%"); a.set_xticks(x); a.set_xticklabels(lab)
    a2 = a.twinx(); a2.plot(x, e, "-o", color="#E64B35", mfc="white", mew=1.1, label="单位工作量电耗")
    for i in range(3): a2.text(x[i], e[i] - 0.017, f"{e[i]:.3f}", fontsize=8, color="#E64B35", ha="center", va="top",
                               bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    a2.set_ylabel("单位工作量电耗/(kWh·GPU$^{-1}$·h$^{-1}$)"); a2.set_ylim(1.0, 1.25)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8); panel(a, "(a)")
    a.text(0.02, 0.03, f"休眠态: 速率 0, 整机待机 {P['P_OFF'] * 1000:.0f} W/台 (运行功率的 {P['P_OFF'] / ptot[0] * 100:.1f}%)\n空闲运行功率 {P['P_SLEEP'] * 1000:.0f} W/台",
           transform=a.transAxes, fontsize=7, va="bottom", bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2))
    zcm = LinearSegmentedColormap.from_list("z", ["#DCEAF7", "#FBE3C8", "#EE9A86"])
    for j, (lo, hi) in enumerate(P["ZONES"]):
        b.axhspan(lo, hi, color=zcm(j / (len(P["ZONES"]) - 1)), lw=0); b.axhline(lo, color="white", lw=0.4)
        b.text(2.62, (lo + hi) / 2, f"$j$={j + 1}", va="center", fontsize=6.5)
    b.bar(x, tsafe - 20, 0.5, bottom=20, color=["#3C5488", "#4DBBD5", "#00A087"], alpha=0.9)
    for i in range(3): b.text(x[i], tsafe[i] + 0.5, f"{tsafe[i]:.1f} ℃", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(lab); b.set_xlim(-0.5, 3.1); b.set_ylim(20, 56)
    b.set_ylabel("允许的液冷温度上限/℃"); panel(b, "(b)")
    b.set_title("$T^{\\mathrm{chip,max}}-R^{\\mathrm{th}}h_{k}$", fontsize=9, loc="left")
    save(fig, "图2_GPU运行模式特性与热安全上限")


# ====================== 图3 液冷温度 × 环境温度: 自然冷却可用系数与可用 GPU 模式 ======================
def fig3():
    P = C.params(); D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    Ta = np.linspace(-15, 44, 300); Tw = np.linspace(25, 53, 300)
    phi = np.clip((Tw[:, None] - P["DT_CDU"] - Ta[None, :]) / P["ITD_DES"], 0, 1)
    cm = LinearSegmentedColormap.from_list("phi", ["#F7F7F7", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    fig = plt.figure(figsize=(W2, 7.0 * CM), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4.2], width_ratios=[1, 0.03])
    ax = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=ax); cax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(Ta, Tw, phi, cmap=cm, shading="auto", vmin=0, vmax=1, rasterized=True)
    cs = ax.contour(Ta, Tw, phi, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    names = {"k1_标称": "k1 标称上限", "k2_浅锁频": "k2 浅锁频上限", "k3_深锁频": "k3 深锁频上限"}
    for k, ls in zip(P["MODES"], ["-", "--", ":"]):
        t = P["T_CHIP_MAX"] - P["R_TH"] * P["MODES"][k][1]
        ax.axhline(t, color="#E64B35", lw=1.1, ls=ls)
        ax.text(-14.5, t - 0.4, names[k] + f" {t:.1f}℃", color="white", fontsize=8, va="top")
    for lo, hi in P["ZONES"][1:]: ax.axhline(lo, color="0.35", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel("室外干球温度 $T^{\\mathrm{amb}}$/℃"); ax.set_ylabel("液冷环路温度 $T^{w}$/℃")
    cb = fig.colorbar(im, cax=cax); cb.set_label("自然冷却可用系数 $\\underline{\\phi}$")
    Tall = D["Tamb_all"].ravel(); hot = np.array([max(b["Tamb"].max() for b in S) for S in D["test"]])
    top.hist(Tall, bins=np.arange(-15, 43, 1.5), color="#9ECAE1", edgecolor="white", lw=0.3)
    top.set_ylabel("小时数"); top.tick_params(labelbottom=False); top.spines[["top", "right"]].set_visible(False)
    for t, lab, ha, dx in ((36.7, "训练集最高\n36.7℃", "right", -0.3), (40.0, "测试集最高\n40.0℃", "left", 0.3)):
        ax.axvline(t, color="k", lw=0.7, ls="--"); ax.text(t + dx, 26, lab, ha=ha, fontsize=7.5)
    ax.set_xlim(-15, 44)
    save(fig, "图3_液冷温区与气象共同决定可用GPU模式和自然冷却能力")


# ====================== 公共工具 ======================
def plan(name): return pickle.load(open(RES / f"plan_{name}.pkl", "rb"))


def hf(name): return pd.read_csv(RES / f"hf_{name}.csv")


def hourly(name, sel=None):
    """统一评价器下的逐时结果 (缓存)。sel: [(d, i)] 测试日; None 表示全部测试日。"""
    import exp as X
    f = RES / f"hourly_{name}.pkl"
    cache = pickle.load(open(f, "rb")) if f.exists() else {}
    R = plan(name); D = X.dataset(X.PLANS[name].get("ds", "base"))
    sel = sel or [(d, i) for d in range(4) for i in range(len(D["test"][d]))]
    new = False
    for d, i in sel:
        if (d, i) not in cache:
            cache[(d, i)] = X.eval_day(R["x"], D["test"][d][i], D["price"], cfg=R["cfg"])[1]; new = True
    if new: pickle.dump(cache, open(f, "wb"))
    return {k: cache[k] for k in sel}


def hottest_test_day():
    D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    best = max(((d, i) for d in range(4) for i in range(len(D["test"][d]))), key=lambda k: D["test"][k[0]][k[1]]["Tamb"].max())
    return best, D


def grad(c, n):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(c)); return [tuple(1 - (1 - c) * f) for f in np.linspace(0.45, 1.0, n)]


E1 = ["M0", "M1", "M2", "M3", "M0-W45"]


# ====================== 图4 E1 容量配置 + 统一复核费用构成 ======================
def fig4():
    s = S(); ms = [m for m in E1 if m in s.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.6 * CM), layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    devs = [k for k in ["PV", "FC", "MC", "EB"] if s.loc[ms, k].abs().max() > 1]; x = np.arange(len(devs)); w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        v = [max(s.loc[m, k], 0) for k in devs]
        a.bar(x - 0.4 + w * (j + 0.5), v, w, color=COLM[m], edgecolor="white", lw=0.4, label=LABM[m])
    a.set_xticks(x); a.set_xticklabels([f"{DEV[k][0]}\n({'kWp' if k == 'PV' else 'kWh' if k == 'EB' else 'kW$_{th}$'})" for k in devs])
    a.set_ylabel("配置容量"); a.set_ylim(0, s.loc[ms, devs].values.max() * 1.12); panel(a, "(a)", -0.16)
    h, l = a.get_legend_handles_labels(); legend_top(fig, h, l, 5, 1.0)
    parts = [("年化_PV", "光伏投资", "#F6C85F"), ("年化_FC", "自然冷却投资", "#7FC97F"), ("年化_MC", "机械制冷投资", "#F4A6A6"),
             ("复核年运行费用", "复核运行费用", "#C9B3DE")]
    others = [m for m in ms if m != "M0"]; xb = np.arange(len(others))
    pos, neg = np.zeros(len(others)), np.zeros(len(others))
    for col, lab, c in parts:
        v = np.array([s.loc[m, col] - s.loc["M0", col] for m in others]) / WAN
        bot = np.where(v >= 0, pos, neg); b.bar(xb, v, 0.55, bottom=bot, color=c, edgecolor="white", lw=0.4, label=lab)
        pos += np.clip(v, 0, None); neg += np.clip(v, None, 0)
    tot = np.array([s.loc[m, "复核能源相关费用"] - s.loc["M0", "复核能源相关费用"] for m in others]) / WAN
    b.scatter(xb, tot, marker="D", s=26, color="k", zorder=4, label="能源相关费用合计变化")
    for i, t_ in enumerate(tot): b.text(i + 0.32, t_, f"{t_:+.1f}", va="center", fontsize=8)
    b.axhline(0, color="0.3", lw=0.7); b.set_xticks(xb); b.set_xticklabels(others); b.set_xlim(-0.6, len(others) - 0.2)
    b.set_ylabel("相对 M0 的年化费用变化/万元"); panel(b, "(b)", -0.13)
    cool = np.array([s.loc[m, "实时失约_冷却热安全受限"] for m in others])
    lo, hi = min(neg.min(), tot.min()), max(pos.max(), 0); b.set_ylim(lo * 1.55, hi + (hi - lo) * 0.12)
    for i, c_ in enumerate(cool):
        b.text(i, lo * 1.45, f"冷却失约\n{c_:.0f} GPU·h", ha="center", va="bottom", fontsize=7.2,
               color="#B03A2E" if c_ > 1000 else ("#E67E22" if c_ > 1 else "#2E8B57"))
    h1, l1 = b.get_legend_handles_labels(); b.legend(h1, l1, ncol=3, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "图4_E1物理机制消融_容量配置与统一复核费用")


# ====================== 图5 两因素瀑布图 (边际贡献与交互项) ======================
def fig5():
    s = S()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 6.0 * CM), layout="constrained")
    for ax, col, ttl in ((axs[0], "规划能源相关费用", "(a) 规划目标 (训练情景)"), (axs[1], "复核能源相关费用", "(b) 统一物理复核 (封存测试日)")):
        c0, c1, c2, c3 = (s.loc[m, col] / WAN for m in ("M0", "M1", "M2", "M3"))
        steps = [("M0", c0, "base"), ("+GPU调节\n(M1−M0)", c1 - c0, "d"), ("+热反馈\n(M2−M0)", c2 - c0, "d"),
                 ("交互项\n$-I_{GT}$", c3 - c1 - c2 + c0, "d"), ("M3", c3, "base")]
        cur = 0; xs = np.arange(len(steps)); lo = min(c0, c3) * 0.9
        for i, (lab, v, kind) in enumerate(steps):
            if kind == "base":
                ax.bar(i, v - lo, 0.6, bottom=lo, color=COLM[lab], edgecolor="white"); cur = v
                ax.text(i, v + (c0 - lo) * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                c = "#2E8B57" if v < 0 else "#C0392B"
                ax.bar(i, v, 0.6, bottom=cur, color=c, alpha=0.85, edgecolor="white")
                ax.text(i, max(cur, cur + v) + (c0 - lo) * 0.02, f"{v:+.1f}", ha="center", va="bottom", fontsize=8, color=c)
                ax.plot([i - 1.3, i + 0.3], [cur, cur], color="0.5", lw=0.5, ls=":"); cur += v
        ax.set_xticks(xs); ax.set_xticklabels([x[0] for x in steps], fontsize=8); ax.set_ylim(lo, max(c0, c3) * 1.04)
        ax.set_ylabel("能源相关年化费用/万元 (纵轴未从0起)"); ax.set_title(ttl, fontsize=9, loc="left")
        I = c1 + c2 - c0 - c3
        ax.text(0.98, 0.95, f"$I_{{GT}}$ = {I:+.1f} 万元", transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(fc="#FFF5E6", ec="#E0A458", lw=0.6, boxstyle="round,pad=0.3"))
    save(fig, "图5_GPU调节与热反馈的边际贡献及交互效应")


# ====================== 图6 活跃约束: 温区运行小时与 GPU 模式结构 ======================
def fig6():
    ms = [m for m in ("M0", "M2", "M3", "U3") if (RES / f"plan_{m}.pkl").exists()]
    D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    fig = plt.figure(figsize=(W2, 6.2 * CM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1]); a = fig.add_subplot(gs[0])
    w = 0.8 / len(ms); x = np.arange(4)
    for j, m in enumerate(ms):
        H = hourly(m); E = np.zeros(4); F = np.zeros(4)
        for (d, i), o in H.items():
            wt = D["n_d"][d] / len(D["test"][d]); E[d] += wt * o["qMC"].sum() / 1e3; F[d] += wt * o["qFC"].sum() / 1e3
        share = E / (E + F) * 100
        a.bar(x - 0.4 + w * (j + 0.5), E, w, color=COLM[m], edgecolor="white", lw=0.4, label=LABM[m])
        for i in range(4):
            if E[i] > 1: a.text(x[i] - 0.4 + w * (j + 0.5), E[i] + 8, f"{share[i]:.0f}%", ha="center", fontsize=6.5, rotation=90)
    a.set_xticks(x); a.set_xticklabels(D["classes"]); a.set_ylabel("机械制冷散热/MWh$_{th}$·年$^{-1}$ (统一复核)")
    a.set_ylim(0, a.get_ylim()[1] * 1.12); a.legend(ncol=2, fontsize=7.2, loc="upper left"); panel(a, "(a)", -0.16, 1.01)
    a.text(0.98, 0.95, "柱顶数字: 机械制冷\n散热占比", transform=a.transAxes, fontsize=7, color="0.4", va="top", ha="right")
    R = plan("M3"); md = {}
    for bb, o in zip(R["blocks"], R["out"]):
        for k, v in o["modes"].items(): md[k] = md.get(k, 0) + bb["p"] * R["n_d"][bb["d"]] * v.sum()
    for k, (r, ttl, pl) in enumerate((("R", "实时任务 GPU·h", "(b)"), ("B", "可延期任务 GPU·h", "(c)"))):
        ax = fig.add_subplot(gs[k + 1]); kk = [q for q in md if q.startswith(r)]
        v = np.clip(np.array([md[q] for q in kk]), 0, None); cols = ["#3C5488", "#4DBBD5", "#00A087"][:len(kk)]
        wd, _ = ax.pie(v, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white"))
        for wdg, vv in zip(wd, v / v.sum()):
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
            if vv > 0.03: ax.text(0.81 * np.cos(ang), 0.81 * np.sin(ang), f"{vv * 100:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(0, 0, ttl.replace(" ", "\n"), ha="center", va="center", fontsize=8)
        ax.legend(wd, [q.split("|")[1].replace("_", " ") for q in kk], fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
        panel(ax, pl, -0.05, 1.0); ax.set_title("M3 规划情景", fontsize=8, y=0.98)
    save(fig, "图6_活跃约束_机械制冷散热与GPU模式结构")


# ====================== 图7 最高温测试日运行调度 (M0 容量 vs M3 容量, 统一评价器) ======================
def fig7():
    (d, i), D = hottest_test_day(); price = D["price"]
    ms = ["M0", "M3"]; H = {m: hourly(m, [(d, i)])[(d, i)] for m in ms}
    fig, axs = plt.subplots(2, 2, figsize=(W2, 9.0 * CM), layout="constrained", sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    t = np.arange(24); tt = np.arange(25)
    ymax = max((H[m]["Pg"] + H[m]["PPV"] + H[m]["Pdis"]).max() for m in ms) * 1.15
    hmax = max((H[m]["qFC"] + H[m]["qMC"]).max() for m in ms) * 1.3
    for k, m in enumerate(ms):
        o = H[m]; a, b = axs[0, k], axs[1, k]
        st = [o["PPV"], o["Pg"], o["Pdis"]]; labs = ["光伏", "电网购电", "储能放电"]; cs = ["#F6C85F", "#8FA9D6", "#5AB4AC"]
        a.stackplot(tt, *[np.r_[x, x[-1]] for x in st], step="post", colors=cs, labels=labs, alpha=0.95, lw=0)
        L = o["Ptot"] + o["Pch"]
        a.step(tt, np.r_[L, L[-1]], where="post", color="#3C5488", lw=1.1, label="总用电负荷")
        cool = o["PFC"] + o["PMC"]
        a.step(tt, np.r_[cool, cool[-1]], where="post", color="#C0392B", lw=1.1, label="冷源电功率")
        a.set_ylim(0, ymax); a.set_title(f"({'ab'[k]}) {LABM[m]} 容量", fontsize=9, loc="left")
        if k == 0: a.set_ylabel("功率/kW")
        b.stackplot(tt, np.r_[o["qFC"], o["qFC"][-1]], np.r_[o["qMC"], o["qMC"][-1]], step="post", colors=["#7FC97F", "#F4A6A6"],
                    labels=["自然冷却散热", "机械制冷散热"], lw=0)
        heat = 0.75 * o["PIT"]; b.step(tt, np.r_[heat, heat[-1]], where="post", color="#3C5488", lw=1.0, ls="--", label="液冷热负荷 $\\beta P^{\\mathrm{IT}}$")
        b.set_xlim(0, 24); b.set_xticks(range(0, 25, 4)); b.set_xlabel("时段/$t$"); b.set_ylim(0, hmax)
        if k == 0: b.set_ylabel("散热功率/kW$_{th}$")
        bt = b.twinx(); bt.plot(t + 0.5, o["Tamb"], "o-", color="#7F7F7F", ms=2.5, lw=0.8, label="室外温度"); bt.set_ylim(0, 45)
        if k == 1: bt.set_ylabel("室外温度/℃", color="0.4")
        b3 = b.twinx(); b3.bar(t + 0.5, o["sR"], 0.7, color="#C0392B", alpha=0.45, label="实时失约")
        b3.set_ylim(0, max(1.0, max(H[mm]["sR"].max() for mm in ms)) * 1.6); b3.set_yticks([])
        tot = o["sR"].sum(); b.text(0.3, hmax * 0.93, f"实时失约 {tot:.0f} GPU·h; 机械制冷电量 {o['PMC'].sum():.0f} kWh", fontsize=7.5)
    h, l = axs[0, 0].get_legend_handles_labels(); legend_top(fig, h, l, 5)
    h, l = axs[1, 0].get_legend_handles_labels(); h3, l3 = b3.get_legend_handles_labels(); h4, l4 = bt.get_legend_handles_labels()
    fig.legend(h + h4 + h3, l + l4 + l3, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False)
    save(fig, "图7_测试集最高温日运行调度_静态基准与完整耦合")


# ====================== 图8 测试日 × 时段 液冷温度热力图 (完整耦合 M3) ======================
def fig8():
    D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb")); H0, H3 = hourly("M0"), hourly("M3")
    keys = sorted(H3, key=lambda k: D["test"][k[0]][k[1]]["w"])
    Ta = np.array([H3[k]["Tamb"] for k in keys]); Q0 = np.array([H0[k]["qMC"] for k in keys]); Q3 = np.array([H3[k]["qMC"] for k in keys])
    lab = [f"{D['month'][D['test'][k[0]][k[1]]['w']]}月" for k in keys]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 7.2 * CM), layout="constrained", sharey=True)
    cm1 = LinearSegmentedColormap.from_list("t", ["#313695", "#74ADD1", "#FFFFBF", "#F46D43", "#A50026"])
    cm2 = LinearSegmentedColormap.from_list("q", ["#FFFFFF", "#FDD0A2", "#F16913", "#7F2704"])
    vmax = max(Q0.max(), Q3.max())
    ims = [axs[0].imshow(Ta, aspect="auto", cmap=cm1, vmin=-15, vmax=40, extent=[0, 24, len(keys), 0]),
           axs[1].imshow(Q0, aspect="auto", cmap=cm2, vmin=0, vmax=vmax, extent=[0, 24, len(keys), 0]),
           axs[2].imshow(Q3, aspect="auto", cmap=cm2, vmin=0, vmax=vmax, extent=[0, 24, len(keys), 0])]
    ttl = ["(a) 室外温度", f"(b) M0 容量: 机械制冷散热\n全年 {Q0.sum() / len(keys) * 365 / 1e3:.0f} MWh$_{{th}}$",
           f"(c) M3 容量: 机械制冷散热\n全年 {Q3.sum() / len(keys) * 365 / 1e3:.0f} MWh$_{{th}}$"]
    for ax, t_ in zip(axs, ttl): ax.set_xticks(range(0, 25, 6)); ax.set_xlabel("时段/$t$"); ax.set_title(t_, fontsize=8.5, loc="left")
    fig.colorbar(ims[0], ax=axs[0], label="℃", pad=0.01); fig.colorbar(ims[2], ax=axs[1:], label="kW$_{th}$", pad=0.01)
    ticks = [j for j in range(len(keys)) if j == 0 or lab[j] != lab[j - 1]]
    axs[0].set_yticks(np.array(ticks) + 0.5); axs[0].set_yticklabels([lab[j] for j in ticks]); axs[0].set_ylabel("封存测试日")
    save(fig, "图8_封存测试日机械制冷散热热力图")


# ====================== 图9 同容量运行比较 (x_ref = M3) ======================
def fig9():
    f = RES / "ops_summary.csv"
    if not f.exists(): return
    O = pd.read_csv(f).set_index("策略"); ps = list(O.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.0 * CM), layout="constrained")
    cols = [COLM[p[:2]] for p in ps]; x = np.arange(len(ps))
    v = O["复核年运行费用"].values / WAN; bad = O["不可安全运行日数"].values
    a.bar(x, v, 0.6, color=cols, edgecolor="white", hatch=["///" if k > 0 else "" for k in bad])
    for i in range(len(ps)):
        a.text(i, v[i] + 1.2, f"{v[i]:.1f}", ha="center", fontsize=8)
        if bad[i] > 0: a.text(i, v[i] * 0.5, f"{bad[i]}天不可\n安全运行\n(未计费用)", ha="center", va="center", fontsize=7, color="#B03A2E",
                              bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    a.set_ylim(0, v.max() * 1.12); a.set_ylabel("年运行费用/万元"); a.set_xticks(x); a.set_xticklabels(ps); panel(a, "(a)", -0.14)
    r = O["实时失约GPUh"].values
    b.bar(x, r, 0.6, color=cols, edgecolor="white", hatch=["///" if k > 0 else "" for k in bad])
    for i in range(len(ps)):
        b.text(i, r[i] + r.max() * 0.02, f"SLA {O['实时SLA'].iloc[i] * 100:.2f}%", ha="center", fontsize=7.5)
    b.set_ylabel("测试日实时任务失约/GPU·h"); b.set_xticks(x); b.set_xticklabels(ps); b.set_ylim(0, r.max() * 1.15); panel(b, "(b)", -0.14)
    fig.text(0.5, -0.07, "设备容量统一取 M3 规划值; 斜线: 存在即使允许失约也无法满足热安全/功率平衡的测试日", ha="center", fontsize=7.5, color="0.35")
    save(fig, "图9_同容量运行比较_只改变运行策略")


# ====================== 图10 不确定性模型: 样本外成本—服务风险前沿 ======================
def fig10():
    s = S(); fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.8 * CM), layout="constrained")
    fam = {"U1": "U1 无条件对称", "U2": "U2 条件对称", "U3": "U3 条件半方差(本文)"}
    norm = Normalize(0.5, 2.0); cmap = plt.get_cmap("viridis")
    for ax, ycol, ylab in ((a, "实时失约_冷却热安全受限", "冷却/热安全导致的实时失约/GPU·h"), (b, "日费用CVaR95", "日运行费用 CVaR$_{0.95}$/元")):
        for u, mk in zip(fam, ["s", "^", "o"]):
            pts = [(k, f"{u}-k{k:g}" if k != 1.0 else u) for k in (0.5, 1.0, 2.0)]
            pts = [(k, n) for k, n in pts if n in s.index]
            xs = [s.loc[n, "复核年运行费用"] / WAN for _, n in pts]; ys = [s.loc[n, ycol] for _, n in pts]
            ax.plot(xs, ys, "-", color=COLM[u], lw=1.0, alpha=0.9)
            sc = ax.scatter(xs, ys, c=[k for k, _ in pts], cmap=cmap, norm=norm, marker=mk, s=40, edgecolor=COLM[u], lw=1.2, zorder=3, label=fam[u])
        ax.scatter(s.loc["M3", "复核年运行费用"] / WAN, s.loc["M3", ycol], marker="D", s=40, color=COLM["M3"], zorder=4, label="U-SP 经验分布")
        ax.scatter(s.loc["U3", "复核年运行费用"] / WAN, s.loc["U3", ycol], s=170, facecolor="none", edgecolor="#E64B35", lw=1.3, zorder=5)
        ax.set_xlabel("封存测试日年运行费用/万元"); ax.set_ylabel(ylab)
    u0 = s.loc["U0"]
    a.text(0.36, 0.42, f"U0 确定性均值未画出: 仅 {u0['N']:.0f} 台GPU,\n实时SLA {u0['实时SLA'] * 100:.1f}%, 设计日失约严重", transform=a.transAxes,
           va="top", fontsize=7.3, color="#B03A2E", bbox=dict(fc="#FDEDEC", ec="none", pad=1.5))
    b.legend(fontsize=7.5, loc="upper right"); panel(a, "(a)", -0.16); panel(b, "(b)", -0.16)
    cb = fig.colorbar(sc, ax=[a, b], pad=0.01, shrink=0.9); cb.set_label("模糊集缩放 κ")
    save(fig, "图10_不确定性模型样本外成本与服务风险前沿")


# ====================== 图11 经验概率与最坏分布 (U3) ======================
def fig11():
    R = plan("U3"); D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    ns = max(len(S_) for S_ in D["sup"]); pad = lambda v: np.r_[v, np.full(ns - len(v), np.nan)]
    P0 = np.array([pad([b["p"] for b in S_]) for S_ in D["sup"]]); PW = np.array([pad(v) for v in R["pw"]])
    longest = max(D["sup"], key=len)
    tags = [b["tag"].replace("代表日", "代表").replace("极端:T_max", "最热日").replace("极端:R_peak", "峰值日") for b in longest]
    tags = [t if not t.startswith("设计极端日") else "设计日" for t in tags]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 5.4 * CM), layout="constrained", sharey=True)
    cm = LinearSegmentedColormap.from_list("p", ["#FFFFFF", "#FDD49E", "#FC8D59", "#B30000"])
    dv = PW - P0; lim = np.nanmax(np.abs(dv))
    for ax, M_, ttl, cmap, vmin, vmax in ((axs[0], P0, "(a) 经验概率 $\\hat p$", cm, 0, 0.6), (axs[1], PW, "(b) 最坏分布 $p^*$", cm, 0, 0.6),
                                         (axs[2], dv, "(c) $p^*-\\hat p$", "RdBu_r", -lim, lim)):
        im = ax.imshow(M_, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for (r, c), v in np.ndenumerate(M_):
            if np.isnan(v): continue
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=0, fontsize=7.5); ax.set_title(ttl, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels(D["classes"])
    save(fig, "图11_条件半方差模糊集下的最坏分布")


# ====================== 图12 连续多日压力测试 ======================
def fig12():
    f = RES / "stress_all.csv"
    if not f.exists(): return
    T = pd.read_csv(f); ms = [m for m in ["M0", "M2", "M3", "U0", "U3"] if m in set(T.方案)]
    segs = list(dict.fromkeys(T.压力片段))
    fig = plt.figure(figsize=(W2, 10.5 * CM), layout="constrained"); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    for k, sg in enumerate(segs):
        ax = fig.add_subplot(gs[0, k]); g = T[T.压力片段 == sg].set_index("方案")
        v = [g.loc[m, "实时失约"] + g.loc[m, "延期失约"] if m in g.index else 0 for m in ms]
        ax.bar(range(len(ms)), v, 0.65, color=[COLM[m] for m in ms], edgecolor="white")
        for i, vv in enumerate(v): ax.text(i, abs(vv) + max(v) * 0.02 + 0.5, f"{abs(vv):.0f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, fontsize=8); ax.set_title(f"({'abc'[k]}) {sg}", fontsize=9, loc="left")
        ax.set_ylim(0, max(max(v) * 1.25, 10))
        if k == 0: ax.set_ylabel("72 h 任务失约/GPU·h")
        mg = np.nanmin([g.loc[m, "最小热安全裕量K"] for m in ms if m in g.index])
        ax.text(0.03, 0.97, f"$T^{{\\mathrm{{amb}}}}_{{\\max}}$={g['Tmax'].iloc[0]:.1f}℃\n热安全裕量≥{mg:.1f} K", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        ax.set_ylim(0, max(max(v) * 1.45, 10))
    ax = fig.add_subplot(gs[1, :]); key = [sg for sg in segs if sg.startswith("ST2")][0].split()[0]
    t = np.arange(72)
    for m in [m for m in ("M0", "U3") if m in ms]:
        p = RES / f"stress_{m}_{key}.pkl"
        if not p.exists(): continue
        o = pickle.load(open(p, "rb"))
        ax.step(np.arange(73), np.r_[o["qMC"], o["qMC"][-1]], where="post", color=COLM[m], lw=1.1, label=f"{LABM[m]} 机械制冷散热")
        if o["sR"].sum() > 1: ax.bar(t + 0.5, o["sR"], 0.8, color=COLM[m], alpha=0.45, label=f"{LABM[m]} 实时失约 (GPU·h)")
    at = ax.twinx(); at.plot(t + 0.5, o["Tamb"], color="0.4", lw=0.8, ls=":", label="室外温度"); at.set_ylabel("室外温度/℃"); at.set_ylim(0, 45)
    ax.set_xlim(0, 72); ax.set_xticks(range(0, 73, 6)); ax.set_xlabel("时段/h (ST2 连续 3 日)"); ax.set_ylabel("机械制冷散热/kW$_{th}$ ; 失约/GPU·h")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = at.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, ncol=3, fontsize=7.5, loc="upper right")
    panel(ax, "(d)", -0.05)
    save(fig, "图12_连续多日压力测试_失约与热安全裕量")


# ====================== 图13 气候窗口 (S2): 山脊图 + 3D 柱 ======================
def fig13():
    s = S(); cl = [("丹佛", "base", ""), ("旧金山", "旧金山", "S2-旧金山-"), ("芝加哥", "芝加哥", "S2-芝加哥-"), ("坦帕", "坦帕", "S2-坦帕-")]
    nm = lambda pre, m: (pre + m) if pre else m
    cl = [c for c in cl if nm(c[2], "M3") in s.index]
    dc = pd.read_csv(RES / "design_check.csv").set_index("方案") if (RES / "design_check.csv").exists() else None
    fig = plt.figure(figsize=(W2, 7.2 * CM), layout="constrained"); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 0.9])
    a = fig.add_subplot(gs[0]); from scipy.stats import gaussian_kde
    cols = ["#3C5488", "#4DBBD5", "#00A087", "#E64B35"]; xs = np.linspace(-25, 42, 300)
    for k, (n_, tag, pre) in enumerate(cl):
        T = pickle.load(open(C.PROC / f"dataset_{tag}.pkl", "rb")); Ta = T["Tamb_all"].ravel()
        y = gaussian_kde(Ta)(xs); y = y / y.max() * 0.9; off = len(cl) - 1 - k
        a.fill_between(xs, off, off + y, color=cols[k], alpha=0.75, lw=0); a.plot(xs, off + y, color="white", lw=0.6)
        a.plot([T["design"]["T"]] * 2, [off, off + 0.5], color="k", lw=1.0); a.text(T["design"]["T"] - 0.8, off + 0.55, f"设计日 {T['design']['T']:.1f}℃", ha="right", fontsize=6.8)
    a.set_yticks(range(len(cl))); a.set_yticklabels([c[0] for c in cl][::-1]); a.set_xlabel("逐时室外温度/℃"); a.set_xlim(-25, 42); panel(a, "(a)", -0.2)
    ax = fig.add_subplot(gs[1], projection="3d")
    for i, (n_, tag, pre) in enumerate(cl):
        for j, m in enumerate(["M0", "M3"]):
            z = max(s.loc[nm(pre, m), "MC"], 0)
            ax.bar3d(i - 0.3, j - 0.25, 0, 0.6, 0.5, max(z, 1), color=COLM[m], alpha=0.85, shade=True, edgecolor="0.25", linewidth=0.3)
            ax.text(i, j, z + 120, f"{z:.0f}", fontsize=6.8, ha="center")
    ax.set_xticks(range(len(cl))); ax.set_xticklabels([c[0] for c in cl], fontsize=7.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["M0", "M3"], fontsize=7.5)
    ax.set_title("机械制冷容量 $K^{\\mathrm{MC}}$/kW$_{th}$", fontsize=8.5, pad=0); ax.tick_params(axis="z", labelsize=7); ax.view_init(24, -58)
    ax.set_box_aspect((1.6, 0.8, 0.9)); panel(ax, "(b)", 0.0, 0.98)
    c = fig.add_subplot(gs[2]); dv = [(s.loc[nm(pre, "M3"), "复核能源相关费用"] - s.loc[nm(pre, "M0"), "复核能源相关费用"]) / WAN for _, _, pre in cl]
    c.barh(range(len(cl)), dv, 0.55, color=["#2E8B57" if v < 0 else "#C0392B" for v in dv])
    for i, (v, (n_, tag, pre)) in enumerate(zip(dv, cl)):
        c.text(v + (1 if v >= 0 else -1), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
        if dc is not None:
            f0 = dc.loc[nm(pre, "M0"), "实时失约"] + dc.loc[nm(pre, "M0"), "延期失约"]
            if f0 > 1: c.text(0.02, i - 0.36, f"M0 设计日失约 {f0:.0f} GPU·h", fontsize=6.8, color="#B03A2E", transform=c.get_yaxis_transform())
    c.axvline(0, color="0.3", lw=0.7); c.set_yticks(range(len(cl))); c.set_yticklabels([n_ for n_, _, _ in cl]); c.invert_yaxis()
    c.set_xlabel("M3−M0 能源相关费用/万元"); lim = max(abs(min(dv)), abs(max(dv))) * 1.6; c.set_xlim(-lim, lim); panel(c, "(c)", -0.2)
    save(fig, "图13_气候窗口对冷源配置与费用的影响")


# ====================== 图14 S1/S3 与温区精度 ======================
def fig14():
    s = S(); fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 5.8 * CM), layout="constrained")
    for m, mk in (("M0", "s"), ("M3", "o")):
        xs, ys, ns = [], [], []
        for f in (0.6, 1.0, 1.4):
            n = m if f == 1.0 else f"S1-aN{f:g}-{m}"
            if n in s.index: xs.append(f); ys.append(s.loc[n, "复核能源相关费用"] / WAN); ns.append(s.loc[n, "N"])
        a.plot(xs, ys, mk + "-", color=COLM[m], mfc="white", mew=1.1, label=LABM[m])
    a.set_xlabel("GPU 年化投资倍率"); a.set_ylabel("能源相关年化费用/万元"); a.legend(fontsize=8); panel(a, "(a)", -0.14)
    for m, mk in (("M0", "s"), ("M3", "o")):
        xs, ys = [], []
        for sh, n in ((15, f"S3-b15-{m}"), (28.6, m), (50, f"S3-b50-{m}")):
            if n in s.index: xs.append(sh); ys.append(s.loc[n, "复核能源相关费用"] / WAN)
        b.plot(xs, ys, mk + "-", color=COLM[m], mfc="white", mew=1.1, label=LABM[m])
    b.axvspan(46, 52, color="#FADBD8", lw=0); b.text(49, b.get_ylim()[0] + 0.5 * (b.get_ylim()[1] - b.get_ylim()[0]), "50%: 机位上限\n下不可行", ha="center", fontsize=7.2, color="#B03A2E")
    b.set_xlim(12, 53); b.set_xlabel("可延期任务占比/%"); b.set_ylabel("能源相关年化费用/万元"); b.legend(fontsize=8, loc="upper center"); panel(b, "(b)", -0.14)
    a.text(0.02, 0.04, "两方案 GPU 均为 2099 台, 容量不随 GPU 价格改变", transform=a.transAxes, fontsize=7.2, color="0.35")
    save(fig, "图14_GPU投资与任务结构敏感性")


# ====================== 图15 E5 求解效率 ======================
def fig15():
    f = RES / "bench_all.csv"
    if not f.exists(): return
    B = pd.read_csv(f); fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 5.8 * CM), layout="constrained")
    sizes = sorted(B["情景数"].unique())
    for k, (alg, c) in enumerate((("A0 对偶单一MILP", "#3C5488"), ("A1 概率割平面", "#E64B35"))):
        for i, n in enumerate(sizes):
            g = B[(B.算法 == alg) & (B.情景数 == n)]
            if g.empty: continue
            r = g.iloc[0]; done = str(r["状态"]) in ("Optimal", "收敛"); x = i + (k - 0.5) * 0.36
            a.bar(x, r["时间s"] / 60, 0.36, color=c, alpha=1 if done else 0.45, hatch="" if done else "///", edgecolor="white",
                  label=alg if i == 0 else None)
            lab = f"{r['时间s'] / 60:.0f} min" if done else (f"间隙{r['绝对间隙'] / 1e4:.0f}万" if r["状态"] == "Time limit reached" else f">{r['时间s'] / 60:.0f} min\n未完成")
            if alg.startswith("A1") and done: lab += f"\n{int(r['迭代'])}次迭代"
            a.text(x, r["时间s"] / 60 * 1.12, lab, ha="center", fontsize=6.8)
    a.set_yscale("log"); a.set_ylim(1, 400); a.set_xticks(range(len(sizes))); a.set_xticklabels([f"{n} 情景" for n in sizes])
    a.set_ylabel("求解时间/min"); a.legend(fontsize=8, loc="upper left"); panel(a, "(a)", -0.14)
    p_ = RES / "bench_hist_1.csv"
    if p_.exists():
        h = pd.read_csv(p_)
        b.plot(h["迭代"], h["主问题下界"] / 1e4, "o-", color="#3C5488", mfc="white", label="主问题下界")
        b.plot(h["迭代"], h["当前解上界"] / 1e4, "s--", color="#E64B35", mfc="white", label="当前解上界")
        r0 = B[(B.算法.str.startswith("A0")) & (B.情景数 == sizes[0])].iloc[0]
        b.axhline(r0["目标值"] / 1e4, color="0.4", ls=":", lw=0.9, label="直接对偶 MILP 目标")
        b.set_xlabel("迭代次数"); b.set_ylabel("目标值/万元"); b.set_xticks(h["迭代"]); b.legend(fontsize=8); panel(b, "(b)", -0.14)
        b.set_title("6 情景算例的割平面收敛过程", fontsize=8.5, loc="left")
    save(fig, "图15_直接对偶求解与概率割平面效率")


# ====================== 图16 年能量流桑基图 (M0 vs M3) ======================
def _sankey(ax, left, right, flows, colors, title):
    """简易桑基图: left/right 为节点名列表, flows[(l, r)] = 值。"""
    tot = sum(flows.values()); gap = tot * 0.04
    def pos(nodes, side):
        y, out = 0, {}
        for n in nodes:
            v = sum(f for (l, r), f in flows.items() if (l if side == 0 else r) == n); out[n] = [y, y + v]; y += v + gap
        return out
    L, R = pos(left, 0), pos(right, 1); lc, rc = {k: v[0] for k, v in L.items()}, {k: v[0] for k, v in R.items()}
    for (l, r), f in flows.items():
        if f <= 0: continue
        y0, y1 = lc[l], rc[r]; lc[l] += f; rc[r] += f
        verts = [(0.08, y0), (0.5, y0), (0.5, y1), (0.92, y1), (0.92, y1 + f), (0.5, y1 + f), (0.5, y0 + f), (0.08, y0 + f), (0.08, y0)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MPath(verts, codes), fc=colors[l], ec="none", alpha=0.45))
    for nodes, side, x in ((L, 0, 0.0), (R, 1, 0.92)):
        for n, (a_, b_) in nodes.items():
            c = colors.get(n, "#7F7F7F"); ax.add_patch(mpatches.Rectangle((x, a_), 0.08, b_ - a_, fc=c, ec="none"))
            ax.text(x - 0.01 if side == 0 else x + 0.09, (a_ + b_) / 2, f"{n}\n{(b_ - a_) / 1e3:.2f} GWh$_{{th}}$", ha="right" if side == 0 else "left",
                    va="center", fontsize=7.5)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-gap, max(max(v[1] for v in L.values()), max(v[1] for v in R.values())) + gap)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=9)


def fig16():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 6.6 * CM), layout="constrained")
    D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb")); nd = D["n_d"]; P = C.params()
    for ax, m, pl in zip(axs, ["M0", "M3"], "ab"):
        H = hourly(m); g = {k: 0.0 for k in ("FC", "MC", "AIR", "eFC", "eMC")}
        for (d, i), o in H.items():
            w = nd[d] / len(D["test"][d])
            g["FC"] += w * o["qFC"].sum(); g["MC"] += w * o["qMC"].sum(); g["AIR"] += w * (1 - P["BETA_LIQ"]) * o["PIT"].sum()
            g["eFC"] += w * o["PFC"].sum(); g["eMC"] += w * o["PMC"].sum()
        flows = {("液冷环路热量", "自然冷却"): g["FC"] / 1e3, ("液冷环路热量", "机械制冷"): g["MC"] / 1e3, ("空气侧热量", "空气侧冷却"): g["AIR"] / 1e3}
        cols = {"液冷环路热量": "#E64B35", "空气侧热量": "#F39B7F", "自然冷却": "#7FC97F", "机械制冷": "#4DBBD5", "空气侧冷却": "#B09C85"}
        _sankey(ax, ["液冷环路热量", "空气侧热量"], ["自然冷却", "机械制冷", "空气侧冷却"], flows, cols,
                f"({pl}) {LABM[m]}: 年散热流向 (统一复核)\n冷源电量 自然冷却 {g['eFC'] / 1e3:.0f} MWh + 机械制冷 {g['eMC'] / 1e3:.0f} MWh")
    save(fig, "图16_年散热流向桑基图_静态基准与完整耦合")


# ====================== 图17 多指标雷达图 ======================
def fig17():
    s = S(); ms = [m for m in ["M0", "M2", "M3", "U3"] if m in s.index]
    met = [("复核能源相关费用", "能源费用", -1), ("复核冷却电量MWh", "冷却电量", -1), ("复核PUE", "PUE", -1), ("年化_MC", "冷水机组投资", -1),
           ("实时SLA", "实时SLA", 1), ("复核光伏利用率", "光伏利用率", 1)]
    V = np.array([[s.loc[m, c] for c, _, _ in met] for m in ms], float)
    sc = np.zeros_like(V)
    for j, (_, _, sg) in enumerate(met):
        lo, hi = V[:, j].min(), V[:, j].max(); z = (V[:, j] - lo) / (hi - lo + 1e-12); sc[:, j] = 0.2 + 0.8 * (z if sg > 0 else 1 - z)
    ang = np.linspace(0, 2 * np.pi, len(met), endpoint=False); ang = np.r_[ang, ang[0]]
    fig = plt.figure(figsize=(W1, 7.6 * CM), layout="constrained"); ax = fig.add_subplot(projection="polar")
    for m, v in zip(ms, sc):
        ax.plot(ang, np.r_[v, v[0]], "-o", color=COLM[m], ms=3, lw=1.1, label=LABM[m]); ax.fill(ang, np.r_[v, v[0]], color=COLM[m], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([n for _, n, _ in met], fontsize=8); ax.tick_params(axis="x", pad=7); ax.set_yticks([0.2, 0.6, 1.0]); ax.set_yticklabels([])
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    ax.set_title("越靠外越优 (各指标按方案间极差归一化)", fontsize=8, pad=12)
    save(fig, "图17_多指标雷达图")


# ====================== 图18 逐日节省山脊图 (M0 − M3, 统一复核, 同一测试日配对) ======================
def fig18():
    a, b = hf("M0").set_index(["d", "i"]), hf("M3").set_index(["d", "i"]); D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    dlt = (b["日运行费用"] - a["日运行费用"]).reset_index()
    fig, ax = plt.subplots(figsize=(W1, 6.4 * CM), layout="constrained"); from scipy.stats import gaussian_kde
    cols = ["#7FC97F", "#E64B35", "#F6C85F", "#4DBBD5"]; xs = np.linspace(dlt["日运行费用"].min() - 300, dlt["日运行费用"].max() + 300, 300)
    for k in range(4):
        v = dlt[dlt.d == k]["日运行费用"].values; off = 3 - k
        y = gaussian_kde(v)(xs) if len(v) > 2 and v.std() > 1e-6 else np.zeros_like(xs); y = y / (y.max() + 1e-12) * 0.85
        ax.fill_between(xs, off, off + y, color=cols[k], alpha=0.7, lw=0); ax.plot(xs, off + y, color="white", lw=0.6)
        ax.plot(v, np.full_like(v, off + 0.04), "|", color="k", ms=4, mew=0.5)
        ax.text(xs[-1], off + 0.35, f"均值 {v.mean():+.0f} 元/日", ha="right", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.6, ls="--"); ax.set_yticks(range(4)); ax.set_yticklabels(D["classes"][::-1])
    ax.set_xlabel("同一测试日运行费用变化 (M3−M0)/元"); ax.set_xlim(xs[0], xs[-1])
    save(fig, "图18_封存测试日逐日运行费用变化山脊图")


# ====================== 图19 GPU 休眠与降频: IT 功率构成 (典型夏季代表日) ======================
def fig19():
    P = C.params(); fig, axs = plt.subplots(1, 2, figsize=(W2, 6.2 * CM), layout="constrained", sharey=True)
    for ax, m, pl in zip(axs, ["M2", "M3"], "ab"):
        R = plan(m); j = max(range(len(R["blocks"])), key=lambda k: (R["blocks"][k]["d"] == 1) * R["blocks"][k]["p"])
        o, b = R["out"][j], R["blocks"][j]; N = R["x"]["N"]; t = np.arange(25)
        act = sum((P["MODES"][k.split("|")[1]][1] + P["P_HOST"] - P["P_SLEEP"]) * v for k, v in o["modes"].items())
        run = sum(o["modes"].values()); off = o["off"]; warm = N - run - off
        parts = [(run * P["P_SLEEP"], "运行GPU基础功率", "#8FA9D6"), (act, "运行GPU增量功率", "#3C5488"),
                 (warm * P["P_SLEEP"], "空闲常开GPU", "#F4A6A6"), (off * P["P_OFF"], "休眠GPU待机", "#7FC97F")]
        ax.stackplot(t, *[np.r_[v, v[-1]] for v, _, _ in parts], step="post", colors=[c for _, _, c in parts], labels=[l for _, l, _ in parts], lw=0)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.set_xlabel("时段/$t$")
        ax.set_title(f"({pl}) {LABM[m]}: IT 电量 {o['PIT'].sum() / 1e3:.1f} MWh/日", fontsize=9, loc="left")
        a2 = ax.twinx(); a2.plot(t[:-1] + 0.5, b["AR"] + b["AB"], "k--", lw=0.9, label="到达工作量"); a2.set_ylim(0, 4000)
        if pl == "b": a2.set_ylabel("到达工作量/GPU·h"); 
        else: a2.set_yticks([])
    axs[0].set_ylabel("IT 功率/kW"); h, l = axs[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    legend_top(fig, h + h2, l + l2, 5)
    save(fig, "图19_GPU休眠与降频对IT功率构成的影响")


FIGS = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8, "9": fig9, "10": fig10, "11": fig11,
        "12": fig12, "13": fig13, "14": fig14, "15": fig15, "16": fig16, "17": fig17, "18": fig18, "19": fig19}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for k in which: FIGS[k]()
