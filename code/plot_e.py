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
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


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
    zc = ["#DCEAF7", "#BBD7EE", "#FBE3C8", "#F7C3A6", "#EE9A86"]
    for j, (lo, hi) in enumerate(P["ZONES"]):
        b.axhspan(lo, hi, color=zc[j], lw=0); b.text(2.62, (lo + hi) / 2, f"温区{j + 1}", va="center", fontsize=8)
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.4 * CM), layout="constrained", gridspec_kw=dict(width_ratios=[1.2, 1]))
    devs = ["PV", "FC", "MC", "EB"]; x = np.arange(len(devs)); w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        v = [max(s.loc[m, k], 0) for k in devs]
        a.bar(x - 0.4 + w * (j + 0.5), v, w, color=COLM[m], edgecolor="white", lw=0.4, label=LABM[m])
    a.set_xticks(x); a.set_xticklabels([f"{DEV[k][0]}\n({'kWp' if k == 'PV' else 'kWh' if k == 'EB' else 'kW$_{th}$'})" for k in devs])
    a.set_ylabel("配置容量"); a.set_ylim(0, None); panel(a, "(a)", -0.12)
    a.legend(ncol=2, fontsize=8, loc="upper right")
    parts = [("年化_PV", "光伏", "#F6C85F"), ("年化_FC", "自然冷却", "#7FC97F"), ("年化_MC", "机械制冷", "#F4A6A6"),
             ("年化_EB", "储能", "#5AB4AC")]
    xb = np.arange(len(ms)); bot = np.zeros(len(ms))
    for col, lab, c in parts:
        v = np.array([s.loc[m, col] + (s.loc[m, "年化_PB"] if col == "年化_EB" else 0) for m in ms]) / WAN
        b.bar(xb, v, 0.62, bottom=bot, color=c, edgecolor="white", lw=0.4, label=lab); bot += v
    op = np.array([s.loc[m, "复核年运行费用"] for m in ms]) / WAN
    b.bar(xb, op, 0.62, bottom=bot, color="#C9B3DE", edgecolor="white", lw=0.4, label="复核运行费用"); bot += op
    for i, m in enumerate(ms): b.text(i, bot[i] + 8, f"{bot[i]:.0f}", ha="center", fontsize=8)
    b.set_xticks(xb); b.set_xticklabels([m for m in ms]); b.set_ylabel("能源相关年化费用/万元"); b.set_ylim(0, bot.max() * 1.22)
    b2 = b.twinx(); cool = np.array([s.loc[m, "实时失约_冷却热安全受限"] for m in ms])
    b2.plot(xb, cool, "D-", color="#B03A2E", mfc="white", mew=1.1, ms=4.8, label="冷却/热安全导致的实时失约")
    b2.set_ylabel("实时失约/GPU·h"); b2.set_ylim(0, max(cool.max() * 1.4, 10))
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, ncol=2, fontsize=7.5, loc="upper left"); panel(b, "(b)", -0.14)
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
        ax.set_ylabel("能源相关年化费用/万元"); ax.set_title(ttl, fontsize=9, loc="left")
        I = c1 + c2 - c0 - c3
        ax.text(0.98, 0.95, f"$I_{{GT}}$ = {I:+.1f} 万元", transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(fc="#FFF5E6", ec="#E0A458", lw=0.6, boxstyle="round,pad=0.3"))
    save(fig, "图5_GPU调节与热反馈的边际贡献及交互效应")


# ====================== 图6 活跃约束: 温区运行小时与 GPU 模式结构 ======================
def fig6():
    P = C.params(); ms = [m for m in ("M2", "M3", "U3") if (RES / f"plan_{m}.pkl").exists()]
    s = S()
    fig = plt.figure(figsize=(W2, 6.2 * CM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1]); a = fig.add_subplot(gs[0])
    zc = ["#DCEAF7", "#9ECAE1", "#FDD49E", "#FC8D59", "#D7301F"]
    for i, m in enumerate(ms):
        h = np.array([s.loc[m, f"温区{j}小时"] for j in range(5)]); h = h / h.sum() * 100; left = 0
        for j in range(5):
            a.barh(i, h[j], left=left, color=zc[j], edgecolor="white", lw=0.5, label=f"温区{j + 1} {P['ZONES'][j][0]:.0f}–{P['ZONES'][j][1]:.0f}℃" if i == 0 else None)
            if h[j] > 6: a.text(left + h[j] / 2, i, f"{h[j]:.0f}%", ha="center", va="center", fontsize=7.5)
            left += h[j]
    a.set_yticks(range(len(ms))); a.set_yticklabels([LABM[m] for m in ms]); a.set_xlim(0, 100); a.set_xlabel("全年运行小时占比/%")
    a.legend(ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); a.invert_yaxis(); panel(a, "(a)", -0.25, 1.18)
    R = plan("M3"); md = {}
    for b, o in zip(R["blocks"], R["out"]):
        for k, v in o["modes"].items(): md[k] = md.get(k, 0) + b["p"] * R["n_d"][b["d"]] * v.sum()
    for k, (r, ttl, pl) in enumerate((("R", "实时任务 GPU·h", "(b)"), ("B", "可延期任务 GPU·h", "(c)"))):
        ax = fig.add_subplot(gs[k + 1]); kk = [x for x in md if x.startswith(r)]
        v = np.array([md[x] for x in kk]); cols = ["#3C5488", "#4DBBD5", "#00A087"][:len(kk)]
        wd, _ = ax.pie(v, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white"))
        for wdg, vv, x in zip(wd, v / v.sum(), kk):
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
            if vv > 0.03: ax.text(0.81 * np.cos(ang), 0.81 * np.sin(ang), f"{vv * 100:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(0, 0, ttl.replace(" ", "\n"), ha="center", va="center", fontsize=8)
        ax.legend(wd, [x.split("|")[1].replace("_", " ") for x in kk], fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1)
        panel(ax, pl, -0.05, 1.0)
    save(fig, "图6_活跃约束_液冷温区运行小时与GPU模式结构")


# ====================== 图7 最高温测试日运行调度 (M0 容量 vs M3 容量, 统一评价器) ======================
def fig7():
    (d, i), D = hottest_test_day(); price = D["price"]
    ms = ["M0", "M3"]; H = {m: hourly(m, [(d, i)])[(d, i)] for m in ms}
    fig, axs = plt.subplots(2, 2, figsize=(W2, 9.0 * CM), layout="constrained", sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    t = np.arange(24); tt = np.arange(25)
    ymax = max((H[m]["Pg"] + H[m]["PPV"] + H[m]["Pdis"]).max() for m in ms) * 1.15
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
        b.step(tt, np.r_[o["Tw"][:-1], o["Tw"][-2]], where="post", color="#E64B35", lw=1.1, label="液冷温度 $T^w$")
        b.plot(t + 0.5, o["Tamb"], "o-", color="#7F7F7F", ms=3, lw=0.8, label="室外温度")
        b.axhline(45.0, color="#3C5488", ls="--", lw=0.8); b.axhline(50.2, color="#4DBBD5", ls="--", lw=0.8)
        b.text(23.8, 45.4, "k1上限", ha="right", fontsize=7); b.text(23.8, 50.6, "k2上限", ha="right", fontsize=7)
        b.set_ylim(15, 56); b.set_xlim(0, 24); b.set_xticks(range(0, 25, 4)); b.set_xlabel("时段/$t$")
        if k == 0: b.set_ylabel("温度/℃")
        b3 = b.twinx(); b3.bar(t + 0.5, o["sR"], 0.7, color="#C0392B", alpha=0.35, label="实时失约")
        b3.set_ylim(0, max(1.0, max(H[mm]["sR"].max() for mm in ms)) * 1.6); b3.set_yticks([])
        tot = o["sR"].sum(); b.text(0.3, 53, f"实时失约 {tot:.0f} GPU·h; 机械制冷电量 {o['PMC'].sum():.0f} kWh", fontsize=7.5)
    h, l = axs[0, 0].get_legend_handles_labels(); legend_top(fig, h, l, 5)
    h, l = axs[1, 0].get_legend_handles_labels(); h3, l3 = b3.get_legend_handles_labels()
    fig.legend(h + h3, l + l3, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    save(fig, "图7_测试集最高温日运行调度_静态基准与完整耦合")


# ====================== 图8 测试日 × 时段 液冷温度热力图 (完整耦合 M3) ======================
def fig8():
    H = hourly("M3"); D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))
    keys = sorted(H, key=lambda k: D["test"][k[0]][k[1]]["w"])
    Tw = np.array([H[k]["Tw"][:-1] for k in keys]); Ta = np.array([H[k]["Tamb"] for k in keys])
    lab = [f"{D['month'][D['test'][k[0]][k[1]]['w']]}月" for k in keys]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 7.5 * CM), layout="constrained", sharey=True)
    cm1 = LinearSegmentedColormap.from_list("t", ["#313695", "#74ADD1", "#FFFFBF", "#F46D43", "#A50026"])
    i1 = a.imshow(Ta, aspect="auto", cmap=cm1, vmin=-15, vmax=40, extent=[0, 24, len(keys), 0])
    i2 = b.imshow(Tw, aspect="auto", cmap="magma_r", vmin=25, vmax=53, extent=[0, 24, len(keys), 0])
    for ax, im, t, pl in ((a, i1, "室外温度/℃", "(a)"), (b, i2, "液冷温度 $T^w$/℃", "(b)")):
        ax.set_xticks(range(0, 25, 4)); ax.set_xlabel("时段/$t$"); fig.colorbar(im, ax=ax, label=t, pad=0.01); panel(ax, pl, -0.1)
    ticks = [j for j in range(len(keys)) if j == 0 or lab[j] != lab[j - 1]]
    a.set_yticks(np.array(ticks) + 0.5); a.set_yticklabels([lab[j] for j in ticks]); a.set_ylabel("封存测试日")
    save(fig, "图8_封存测试日液冷温度随气象自适应调节热力图")


# ====================== 图9 同容量运行比较 (x_ref = M3) ======================
def fig9():
    f = RES / "ops_summary.csv"
    if not f.exists(): return
    O = pd.read_csv(f).set_index("策略"); ps = list(O.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 5.8 * CM), layout="constrained")
    cols = [COLM[p[:2]] for p in ps]; x = np.arange(len(ps))
    v = O["复核年运行费用"].values / WAN
    a.bar(x, v, 0.6, color=cols, edgecolor="white")
    for i in range(len(ps)): a.text(i, v[i] + v.max() * 0.01, f"{v[i]:.1f}", ha="center", fontsize=8)
    a.set_ylim(v.min() * 0.95, v.max() * 1.04); a.set_ylabel("年运行费用/万元"); a.set_xticks(x); a.set_xticklabels(ps)
    a.text(0.02, 0.96, "纵轴未从0起, 仅放大差异", transform=a.transAxes, fontsize=7, color="0.4", va="top"); panel(a, "(a)", -0.14)
    r = O["实时失约GPUh"].values; bb = O["延期失约GPUh"].values
    b.bar(x - 0.18, r, 0.36, color="#C0392B", label="实时任务失约"); b.bar(x + 0.18, bb, 0.36, color="#E59866", label="可延期任务失约")
    b.set_ylabel("测试日失约/GPU·h"); b.set_xticks(x); b.set_xticklabels(ps); b.legend(fontsize=8); panel(b, "(b)", -0.14)
    b2 = b.twinx(); b2.plot(x, O["复核PUE"].values, "o-", color="#3C5488", mfc="white", label="PUE"); b2.set_ylabel("PUE")
    b2.legend(fontsize=8, loc="upper center")
    save(fig, "图9_同容量运行比较_只改变运行策略")


# ====================== 图10 不确定性模型: 样本外成本—服务风险前沿 ======================
def fig10():
    s = S(); fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.6 * CM), layout="constrained")
    fam = {"U1": "U1 无条件对称", "U2": "U2 条件对称", "U3": "U3 条件半方差(本文)"}
    norm = Normalize(0.5, 2.0); cmap = plt.get_cmap("viridis")
    for ax, ycol, ylab in ((a, "实时失约_冷却热安全受限", "冷却/热安全导致的实时失约/GPU·h"), (b, "日费用CVaR95", "日运行费用 CVaR$_{0.95}$/元")):
        for u, mk in zip(fam, ["s", "^", "o"]):
            pts = [(k, f"{u}-k{k:g}" if k != 1.0 else u) for k in (0.5, 1.0, 2.0)]
            pts = [(k, n) for k, n in pts if n in s.index]
            xs = [s.loc[n, "复核年化总费用"] / WAN for _, n in pts]; ys = [s.loc[n, ycol] for _, n in pts]
            ax.plot(xs, ys, "-", color=COLM[u], lw=0.9, alpha=0.8)
            sc = ax.scatter(xs, ys, c=[k for k, _ in pts], cmap=cmap, norm=norm, marker=mk, s=34, edgecolor=COLM[u], lw=1.0, zorder=3, label=fam[u])
        for n, mk in (("M3", "D"), ("U0", "X")):
            if n in s.index:
                ax.scatter(s.loc[n, "复核年化总费用"] / WAN, s.loc[n, ycol], marker=mk, s=40, color=COLM[n], zorder=4,
                           label={"M3": "U-SP 经验分布", "U0": "U0 确定性均值"}[n])
        if "U3" in s.index:
            ax.scatter(s.loc["U3", "复核年化总费用"] / WAN, s.loc["U3", ycol], s=150, facecolor="none", edgecolor="#E64B35", lw=1.3, zorder=5)
        ax.set_xlabel("统一复核年化总费用/万元"); ax.set_ylabel(ylab)
    a.legend(fontsize=7.5, loc="upper right"); panel(a, "(a)", -0.16); panel(b, "(b)", -0.16)
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
        for i, vv in enumerate(v): ax.text(i, vv + max(v) * 0.02 + 0.5, f"{vv:.0f}", ha="center", fontsize=7.5)
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
        ax.step(np.arange(73), np.r_[o["Tw"][:-1], o["Tw"][-2]], where="post", color=COLM[m], lw=1.1, label=f"{LABM[m]} $T^w$")
        if o["sR"].sum() > 1: ax.bar(t + 0.5, -o["sR"] / 40, 0.8, bottom=22, color=COLM[m], alpha=0.6, label=f"{LABM[m]} 实时失约")
    ax.plot(t + 0.5, o["Tamb"], color="0.4", lw=0.8, ls=":", label="室外温度")
    ax.set_xlim(0, 72); ax.set_xticks(range(0, 73, 6)); ax.set_xlabel("时段/h (ST2 连续 3 日)"); ax.set_ylabel("温度/℃"); ax.set_ylim(5, 56)
    ax.text(0.5, 21.3, "柱: 实时失约 (向下, 1格≈40 GPU·h)", fontsize=7.5, va="top")
    ax.legend(ncol=3, fontsize=7.5, loc="lower right"); panel(ax, "(d)", -0.05)
    save(fig, "图12_连续多日压力测试_失约与热安全裕量")


# ====================== 图13 气候窗口 (S2): 山脊图 + 3D 柱 ======================
def fig13():
    s = S(); cl = [("丹佛(基准)", "base", ""), ("旧金山", "旧金山", "S2-旧金山-"), ("芝加哥", "芝加哥", "S2-芝加哥-"), ("坦帕", "坦帕", "S2-坦帕-")]
    cl = [c for c in cl if (c[2] + "M3" if c[2] else "M3") in s.index]
    fig = plt.figure(figsize=(W2, 7.6 * CM), layout="constrained"); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    a = fig.add_subplot(gs[0]); from scipy.stats import gaussian_kde
    cols = ["#3C5488", "#4DBBD5", "#00A087", "#E64B35"]; xs = np.linspace(-25, 42, 300)
    for k, (nm, tag, pre) in enumerate(cl):
        T = pickle.load(open(C.PROC / f"dataset_{tag}.pkl", "rb"))["Tamb_all"].ravel()
        y = gaussian_kde(T)(xs); y = y / y.max() * 0.9; off = len(cl) - 1 - k
        a.fill_between(xs, off, off + y, color=cols[k], alpha=0.75, lw=0); a.plot(xs, off + y, color="white", lw=0.6)
        hrs = (T >= 34.4).sum(); a.text(41, off + 0.15, f"≥34.4℃ {hrs} h", ha="right", fontsize=7.5)
    a.set_yticks(range(len(cl))); a.set_yticklabels([c[0] for c in cl][::-1]); a.set_xlabel("逐时室外温度/℃"); a.set_xlim(-25, 42)
    a.axvline(34.4, color="k", lw=0.6, ls="--"); panel(a, "(a)", -0.2)
    ax = fig.add_subplot(gs[1], projection="3d")
    nm = lambda pre, m: (pre + m) if pre else m
    xx, yy, zz, cc = [], [], [], []
    for i, (n_, tag, pre) in enumerate(cl):
        for j, m in enumerate(["M0", "M3"]):
            xx.append(i); yy.append(j); zz.append(s.loc[nm(pre, m), "复核能源相关费用"] / WAN); cc.append(COLM[m])
    zz = np.array(zz); cmap = LinearSegmentedColormap.from_list("h", ["#C6DBEF", "#FDD49E", "#E34A33"])
    ax.bar3d(np.array(xx) - 0.3, np.array(yy) - 0.25, 0, 0.6, 0.5, zz, color=cmap(Normalize(zz.min() * 0.9, zz.max())(zz)), shade=True,
             edgecolor="0.3", linewidth=0.3)
    for x_, y_, z_ in zip(xx, yy, zz): ax.text(x_, y_, z_ + zz.max() * 0.04, f"{z_:.0f}", fontsize=6.5, ha="center")
    ax.set_xticks(range(len(cl))); ax.set_xticklabels([c[0].replace("(基准)", "") for c in cl], fontsize=7.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["M0", "M3"], fontsize=7.5); ax.set_zlabel("能源相关费用/万元", fontsize=8, labelpad=2)
    ax.view_init(24, -58); ax.tick_params(axis="z", labelsize=7); ax.set_box_aspect((1.6, 0.8, 0.9)); panel(ax, "(b)", 0.0, 0.95)
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
        for x_, y_, n_ in zip(xs, ys, ns): a.text(x_, y_ + 6, f"N={n_:.0f}", fontsize=7, ha="center", color=COLM[m])
    a.set_xlabel("GPU 年化投资倍率"); a.set_ylabel("能源相关年化费用/万元"); a.legend(fontsize=8); panel(a, "(a)", -0.14)
    for m, mk in (("M0", "s"), ("M3", "o")):
        xs, ys = [], []
        for sh, n in ((15, f"S3-b15-{m}"), (28.6, m), (50, f"S3-b50-{m}")):
            if n in s.index: xs.append(sh); ys.append(s.loc[n, "复核能源相关费用"] / WAN)
        b.plot(xs, ys, mk + "-", color=COLM[m], mfc="white", mew=1.1, label=LABM[m])
    b.set_xlabel("可延期任务占比/%"); b.set_ylabel("能源相关年化费用/万元"); b.legend(fontsize=8); panel(b, "(b)", -0.14)
    save(fig, "图14_GPU投资与任务结构敏感性")


# ====================== 图15 E5 求解效率 ======================
def fig15():
    f = RES / "bench_all.csv"
    if not f.exists(): return
    B = pd.read_csv(f); fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 5.6 * CM), layout="constrained")
    for k, (alg, c) in enumerate((("A0 对偶单一MILP", "#3C5488"), ("A1 概率割平面", "#E64B35"))):
        g = B[B.算法 == alg].sort_values("情景数")
        a.bar(np.arange(len(g)) + (k - 0.5) * 0.36, g["时间s"], 0.36, color=c, label=alg)
        for i, (t_, it) in enumerate(zip(g["时间s"], g["迭代"])):
            a.text(i + (k - 0.5) * 0.36, t_ * 1.15, f"{t_:.0f}s" + (f"\n{it}次" if k else ""), ha="center", fontsize=7)
        a.set_xticks(range(len(g))); a.set_xticklabels([f"{n}情景" for n in g["情景数"]])
    a.set_yscale("log"); a.set_ylabel("求解时间/s"); a.legend(fontsize=8, loc="upper left"); panel(a, "(a)", -0.14)
    for n, c in zip((1, 2, 4), ("#91D1C2", "#4DBBD5", "#3C5488")):
        p = RES / f"bench_hist_{n}.csv"
        if p.exists():
            h = pd.read_csv(p); b.semilogy(h["迭代"], np.maximum(h["违背量"], 1), "o-", color=c, mfc="white", label=f"{n}类 {n * 6}情景")
    b.axhline(1, color="0.5", ls="--", lw=0.6); b.set_xlabel("迭代次数"); b.set_ylabel("最坏分布违背量/元"); b.legend(fontsize=8); panel(b, "(b)", -0.14)
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
            ax.text(x - 0.01 if side == 0 else x + 0.09, (a_ + b_) / 2, f"{n}\n{(b_ - a_) / 1e3:.2f} GWh", ha="right" if side == 0 else "left",
                    va="center", fontsize=7.5)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-gap, max(max(v[1] for v in L.values()), max(v[1] for v in R.values())) + gap)
    ax.invert_yaxis(); ax.axis("off"); ax.set_title(title, fontsize=9)


def fig16():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 7.0 * CM), layout="constrained")
    D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb")); nd = D["n_d"]
    for ax, m, pl in zip(axs, ["M0", "M3"], "ab"):
        H = hourly(m); agg = {k: 0.0 for k in ("PV", "G", "IT", "AUX", "FC", "MC", "P0")}
        for (d, i), o in H.items():
            w = nd[d] / len(D["test"][d]); agg["PV"] += w * o["PPV"].sum(); agg["G"] += w * (o["Pg"].sum() + o["Pdis"].sum() - o["Pch"].sum())
            agg["IT"] += w * o["PIT"].sum(); agg["AUX"] += w * ((o["kaux"] - 1) * o["PIT"]).sum(); agg["FC"] += w * o["PFC"].sum()
            agg["MC"] += w * o["PMC"].sum(); agg["P0"] += w * 30.0 * 24
        tot = agg["PV"] + agg["G"]; dem = {"GPU/IT": agg["IT"], "配电与辅助": agg["AUX"] + agg["P0"], "自然冷却": agg["FC"], "机械制冷": agg["MC"]}
        sc = tot / sum(dem.values()); dem = {k: v * sc for k, v in dem.items()}
        flows = {(s_, k): (agg["PV"] if s_ == "光伏" else agg["G"]) / tot * v for s_ in ("光伏", "电网") for k, v in dem.items()}
        cols = {"光伏": "#F6C85F", "电网": "#8FA9D6", "GPU/IT": "#3C5488", "配电与辅助": "#B09C85", "自然冷却": "#7FC97F", "机械制冷": "#F4A6A6"}
        _sankey(ax, ["光伏", "电网"], list(dem), {k: v / 1e3 for k, v in flows.items()}, cols, f"({pl}) {LABM[m]} (统一复核, 年)")
    save(fig, "图16_年能量流桑基图_静态基准与完整耦合")


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


FIGS = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8, "9": fig9, "10": fig10, "11": fig11,
        "12": fig12, "13": fig13, "14": fig14, "15": fig15, "16": fig16, "17": fig17, "18": fig18}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for k in which: FIGS[k]()
