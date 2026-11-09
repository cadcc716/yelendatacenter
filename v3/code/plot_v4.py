"""
v4 结果图（SCI 单栏样式）：中文宋体、英文数字 Times New Roman（本机用度量兼容替代字体）；坐标标题 10 pt，刻度与图例 9 pt；
主线宽 1.1 pt，Marker 4.8 pt；白底，图例无边框；单栏宽 8.5 cm；600 dpi PNG。每张图只表达一个对比，优先给出相对 Baseline(S1) 的变化。
图数据同时写入 figures_v4/data/*.csv，供 MATLAB 脚本（matlab_figs/）以 SimSun/Times New Roman 重绘。
用法：python plot_v4.py [结果目录]，缺省 results/
"""
import os, sys, glob, pickle
import numpy as np, pandas as pd
import plot_style as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import params as PR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f"{ROOT}/results"
os.environ["RES_DIR"] = RES
import analyze as AN
FIG = f"{ROOT}/figures_v4"; DAT = f"{FIG}/data"; os.makedirs(DAT, exist_ok=True)
PS.style_b(); W = PS.W_SINGLE; CM = PS.CM
SITE = "zhongwei"; P = PR.params()
C = dict(S1="#4D4D4D", S2="#4DBBD5", S3="#00A087", S4="#3C5488", S5="#E64B35", S6="#8491B4", S7="#F39B7F")
SN = dict(S1="S1", S2="S2", S3="S3", S4="S4", S5="S5")


def save(fig, name, df=None):
    fig.savefig(f"{FIG}/{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    if df is not None: df.to_csv(f"{DAT}/{name}.csv", index=False, encoding="utf-8-sig")
    print("已保存", name)


def tab():
    E = AN.table_e(SITE).copy(); E["key"] = E["场景"].str[:2]; return E.set_index("key")


def days(tag, date):
    d = f"{RES}/fig_data" if tag == "DRO_S5" else f"{RES}/fig_data_{tag}"
    f = f"{d}/{SITE}_{date}_逐时.csv"
    if not os.path.exists(f): return None, None
    return pd.read_csv(f), pd.read_csv(f.replace("逐时", "档位"))


def hours(ax, T=24, step=4, label="时刻/h"):
    ax.set_xlim(0, T); ax.set_xticks(range(0, T + 1, step)); ax.set_xlabel(label)


def shade(ax, hrs, color="#9E9E9E", alpha=0.12):
    for h in hrs: ax.axvspan(h, h + 1, color=color, alpha=alpha, lw=0)


# ------------------------------------------------------------------ 图5 能源侧成本逐级分解（相对 S1 的变化量）
def fig_waterfall(E):
    keys = ["S1", "S2", "S3", "S4", "S5"]; v = E.loc[keys, "能源侧年化成本/万元"].values
    step = np.diff(v); lab = ["动态\n调频", "热安全\n耦合", "蓄冷", "余热\n回收"]
    fig, ax = plt.subplots(figsize=(W, 5.6 * CM))
    cum = 0.0; rows = []
    for i, (d, l) in enumerate(zip(step, lab)):
        col = "#00A087" if d < 0 else "#E64B35"
        ax.bar(i, d, bottom=cum, width=0.6, color=col, edgecolor="black", linewidth=0.4)
        ax.text(i, cum + d + (-1 if d < 0 else 1) * 0.04 * abs(v[-1] - v[0]), f"{d:+.1f}\n({100 * d / v[0]:+.1f}%)",
                ha="center", va="top" if d < 0 else "bottom", fontsize=8)
        if i < len(step) - 1: ax.plot([i + 0.3, i + 0.7], [cum + d] * 2, color="gray", lw=0.6, ls="--")
        rows.append(dict(环节=l.replace("\n", ""), 变化量_万元=d, 相对S1_百分比=100 * d / v[0])); cum += d
    ax.bar(len(step), cum, width=0.6, color="#3C5488", edgecolor="black", linewidth=0.4)
    ax.text(len(step), cum - 0.04 * abs(cum), f"{cum:+.1f}\n({100 * cum / v[0]:+.1f}%)", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(step) + 1)); ax.set_xticklabels(lab + ["S5相对\nS1"], fontsize=9)
    ax.set_ylabel("能源侧年化成本变化/万元"); lo = min(np.cumsum(np.r_[0, step]).min(), cum)
    ax.set_ylim(lo * 1.35, max(0, np.cumsum(step).max()) + 0.25 * abs(lo))
    rows.append(dict(环节="S5相对S1", 变化量_万元=cum, 相对S1_百分比=100 * cum / v[0]))
    save(fig, "图5_能源侧成本逐级分解", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图6 冷热源容量（S1–S5，标注相对 S1 变化）
def fig_capacity(E):
    keys = ["S1", "S2", "S3", "S4", "S5"]
    items = [("K_MC/MW", "机械制冷/MW"), ("K_FC/MW", "自然冷却/MW"), ("E_TES/MWh", "蓄冷/MWh"), ("K_HP/MW", "余热热泵/MW")]
    fig, axs = plt.subplots(2, 2, figsize=(W, 7.2 * CM), sharex=True); rows = []
    for ax, (k, yl), tag in zip(axs.ravel(), items, "abcd"):
        v = E.loc[keys, k].values
        ax.bar(range(5), v, 0.62, color=[C[s] for s in keys], edgecolor="black", linewidth=0.4)
        top = max(v.max(), 1e-3)
        for i, x in enumerate(v): ax.text(i, x + 0.03 * top, f"{x:.2f}" if x < 100 else f"{x:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, top * 1.28); ax.set_ylabel(yl); ax.set_title(f"({tag})", fontsize=9, loc="left", pad=2)
        ax.set_xticks(range(5)); ax.set_xticklabels(keys)
        for s, x in zip(keys, v): rows.append(dict(场景=s, 指标=yl, 值=x))
    fig.tight_layout(h_pad=0.4, w_pad=0.6); save(fig, "图6_冷热源容量对比", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图7 关键指标相对 Baseline 的变化率
def fig_kpi(E):
    keys = ["S2", "S3", "S4", "S5"]; base = E.loc["S1"]
    items = [("能源侧年化成本/万元", "能源侧成本"), ("碳排放/t", "碳排放"), ("冷却用电/MWh", "冷却用电"), ("IT用电/MWh", "IT用电")]
    fig, ax = plt.subplots(figsize=(W, 5.8 * CM)); w = 0.19; rows = []
    for j, (k, nm) in enumerate(items):
        d = 100 * (E.loc[keys, k].values / base[k] - 1)
        ax.bar(np.arange(len(keys)) + (j - 1.5) * w, d, w, label=nm, color=["#3C5488", "#00A087", "#4DBBD5", "#F39B7F"][j],
               edgecolor="black", linewidth=0.3)
        for s, x in zip(keys, d): rows.append(dict(场景=s, 指标=nm, 相对S1变化率_百分比=x))
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"{s}相对S1" for s in keys])
    ax.set_ylabel("相对Baseline(S1)变化率/%"); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, handlelength=1.0, columnspacing=0.8)
    pue = E.loc[["S1"] + keys, "名义PUE"].values
    for i, (s, pv) in enumerate(zip(keys, pue[1:])): rows.append(dict(场景=s, 指标="PUE", 相对S1变化率_百分比=100 * (pv / pue[0] - 1)))
    save(fig, "图7_关键指标相对Baseline变化", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图8 夏季典型日 GPU 档位（S1 与 S5）
def fig_gears():
    hs, gs = days("DRO_S5", "2023-06-15"); h1, g1 = days("DRO_S1", "2023-06-15")
    if gs is None: return
    cols = ["#B2182B", "#EF8A62", "#FDDBC7", "#67A9CF", "#2166AC"]; panels = [("S5", gs, hs)] + ([("S1", g1, h1)] if g1 is not None else [])
    fig, axs = plt.subplots(len(panels), 1, figsize=(W, (3.6 * len(panels) + 1.0) * CM), sharex=True); axs = np.atleast_1d(axs); rows = []
    for ax, (s, g, h), tag in zip(axs, panels[::-1], "ab"):
        G = g[[f"档{k}" for k in range(1, 6)]].values; tot = G.sum(1)
        bottom = np.zeros(24)
        for k in range(5):
            ax.bar(np.arange(24) + 0.5, G[:, k] / 1e3, 1.0, bottom=bottom, color=cols[k], edgecolor="none", label=f"档{k + 1}")
            bottom += G[:, k] / 1e3
        a2 = ax.twinx(); a2.plot(np.arange(24) + 0.5, h["液冷温度"], "k-o", ms=2.6, lw=1.0, label="液冷温度")
        a2.set_ylabel("液冷温度/℃"); a2.set_ylim(28, 56)
        ax.set_ylabel("工作GPU/千卡"); ax.set_ylim(0, tot.max() / 1e3 * 1.25); ax.set_title(f"({tag}) {s}", fontsize=9, loc="left", pad=2)
        for t in range(24): rows.append(dict(场景=s, 时刻=t, **{f"档{k + 1}": G[t, k] for k in range(5)}, 液冷温度=h["液冷温度"][t]))
    hours(axs[-1]); h1_, l1_ = axs[0].get_legend_handles_labels()
    axs[0].legend(h1_, l1_, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.12), fontsize=8, handlelength=1.0, columnspacing=0.6)
    fig.tight_layout(h_pad=0.3); save(fig, "图8_夏季典型日GPU档位", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图9 夏季典型日液冷温度与冷机出力（S3、S4、S5）
def fig_thermal():
    hs, _ = days("DRO_S5", "2023-06-15"); h3, _ = days("DRO_S3", "2023-06-15"); h4, _ = days("DRO_S4", "2023-06-15")
    if hs is None: return
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.6 * CM), sharex=True); t = np.arange(24) + 0.5; rows = []
    for s, h, ls in (("S3", h3, "--"), ("S4", h4, ":"), ("S5", hs, "-")):
        if h is None: continue
        axs[0].plot(t, h["液冷温度"], ls, color=C[s], marker="o", ms=2.8, label=s)
        axs[1].plot(t, h["机械制冷"] / 1e3, ls, color=C[s], marker="s", ms=2.8, label=f"{s}机械制冷")
        for i in range(24): rows.append(dict(场景=s, 时刻=i, 液冷温度=h["液冷温度"][i], 机械制冷_kW=h["机械制冷"][i], 蓄冷放冷_kW=h["蓄冷放冷"][i], 气温=h["气温"][i]))
    axs[1].bar(t, hs["蓄冷放冷"] / 1e3, 0.8, color="#3C5488", alpha=0.35, label="S5蓄冷放冷")
    shade(axs[0], P["PEAK_H"]); shade(axs[1], P["PEAK_H"])
    axs[0].set_ylabel("液冷温度/℃"); axs[0].legend(ncol=3, loc="upper left", fontsize=8); axs[0].set_title("(a)", fontsize=9, loc="left", pad=2)
    axs[1].set_ylim(0, axs[1].get_ylim()[1] * 1.35); axs[1].set_ylabel("冷量/MW"); axs[1].legend(ncol=2, loc="upper left", fontsize=8); axs[1].set_title("(b)", fontsize=9, loc="left", pad=2)
    a3 = axs[0].twinx(); a3.plot(t, hs["气温"], color="gray", lw=0.8, alpha=0.7); a3.set_ylabel("气温/℃", color="gray")
    hours(axs[1]); fig.tight_layout(h_pad=0.3); save(fig, "图9_夏季典型日液冷温度与冷源", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图10 可延期任务时移（S5 与 S1 的处理量减到达量）
def fig_shift():
    hs, _ = days("DRO_S5", "2023-06-15"); h1, _ = days("DRO_S1", "2023-06-15")
    if hs is None: return
    fig, ax = plt.subplots(figsize=(W, 5.0 * CM)); t = np.arange(24); rows = []
    w = 0.42 if h1 is not None else 0.8
    for k, (s, h) in enumerate([("S1", h1), ("S5", hs)]):
        if h is None: continue
        d = (h["可延期处理"] - h["可延期到达"]) / 1e3
        ax.bar(t + 0.5 + (k - 0.5) * w * (h1 is not None), d, w, color=C[s], edgecolor="black", linewidth=0.3, label=s)
        for i in range(24): rows.append(dict(场景=s, 时刻=i, 处理减到达_千GPUh=d[i]))
    shade(ax, P["VALLEY_H"], "#4DBBD5", 0.15); ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("处理量−到达量/千GPU·h"); hours(ax); ax.legend(loc="upper center", ncol=2)
    save(fig, "图10_可延期任务时移", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图11 采暖季典型日（S4 与 S5）
def fig_winter():
    date = "2023-11-22"; hs, _ = days("DRO_S5", date); h4, _ = days("DRO_S4", date)
    if hs is None: return
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.2 * CM), sharex=True); t = np.arange(24) + 0.5; rows = []
    for s, h, ls in (("S4", h4, "--"), ("S5", hs, "-")):
        if h is None: continue
        axs[0].plot(t, h["液冷温度"], ls, color=C[s], marker="o", ms=2.8, label=s)
        for i in range(24): rows.append(dict(场景=s, 时刻=i, 液冷温度=h["液冷温度"][i], 热泵供热_kW=h["热泵供热"][i], 热需求_kW=h["热需求"][i]))
    axs[0].axhspan(45, 55, color="#E64B35", alpha=0.08, lw=0); axs[0].set_ylabel("液冷温度/℃"); axs[0].legend(loc="lower right")
    axs[0].set_title("(a)", fontsize=9, loc="left", pad=2)
    axs[1].bar(t, hs["热泵供热"] / 1e3, 0.8, color="#E64B35", alpha=0.8, label="S5热泵供热")
    axs[1].plot(t, hs["热需求"] / 1e3, "k-", lw=1.1, label="热需求"); axs[1].set_ylabel("热功率/MW")
    axs[1].set_ylim(0, hs["热需求"].max() / 1e3 * 1.4); axs[1].legend(loc="upper center", ncol=2)
    axs[1].set_title("(b)", fontsize=9, loc="left", pad=2); hours(axs[1]); fig.tight_layout(h_pad=0.3)
    save(fig, "图11_采暖季典型日余热回收", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图12 忽略热耦合的后果（表F）
def fig_tableF():
    f = f"{RES}/{SITE}_tableF_days.csv"
    if not os.path.exists(f): return
    D = pd.read_csv(f); print(D.columns.tolist()[:12])
    save(plt.figure(), "_tmp")


# ------------------------------------------------------------------ 图13 SP/DRO/RO 样本外测试（相对 DRO 的差值）
def fig_oos():
    import acceptance as AC
    G = {m: AC.oos(SITE, t, None, f"{RES}/ext2", f"{SITE}_ext") for m, t in (("SP", "SP_S6"), ("DRO", "DRO_S5"), ("RO", "RO_S7"))}
    if not all(G.values()): return
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.8 * CM)); m = list(G); rows = []
    d = [G[k]["期望总成本"] - G["DRO"]["期望总成本"] for k in m]
    axs[0].bar(range(3), d, 0.6, color=[C["S6"], C["S5"], C["S7"]], edgecolor="black", linewidth=0.4)
    for i, x in enumerate(d): axs[0].text(i, x + (8 if x >= 0 else -8), f"{x:+.0f}", ha="center", va="bottom" if x >= 0 else "top", fontsize=8)
    axs[0].axhline(0, color="black", lw=0.6); axs[0].set_xticks(range(3)); axs[0].set_xticklabels(m)
    axs[0].set_ylabel("期望年化总成本−DRO/万元"); axs[0].set_title("(a)", fontsize=9, loc="left", pad=2)
    axs[0].set_ylim(min(d) - 0.25 * (max(d) - min(d) + 1), max(d) + 0.3 * (max(d) - min(d) + 1))
    pr = [100 * G[k]["缺额日概率"] for k in m]
    axs[1].bar(range(3), pr, 0.6, color=[C["S6"], C["S5"], C["S7"]], edgecolor="black", linewidth=0.4)
    for i, x in enumerate(pr): axs[1].text(i, x + 0.3, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(m); axs[1].set_ylabel("实时缺额日概率/%"); axs[1].set_ylim(0, max(pr) * 1.3 + 0.5)
    axs[1].set_title("(b)", fontsize=9, loc="left", pad=2)
    for k in m: rows.append(dict(方法=k, **{a: float(b) for a, b in G[k].items()}))
    fig.tight_layout(w_pad=0.8); save(fig, "图13_SP_DRO_RO样本外对比", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图14 C&CG 收敛
def fig_ccg():
    f = f"{RES}/ext2/{SITE}_ext_DRO_S5.pkl"
    if not os.path.exists(f): f = f"{RES}/{SITE}_DRO_S5.pkl"
    if not os.path.exists(f): return
    r = pickle.load(open(f, "rb")); H = pd.DataFrame(r["hist"])
    fig, ax = plt.subplots(figsize=(W, 5.0 * CM)); k = H["it"].values
    ax.plot(k, H["UB"] / 1e4, "o-", color="#E64B35", label="上界"); ax.plot(k, H["LB"] / 1e4, "s-", color="#3C5488", label="下界")
    a2 = ax.twinx(); a2.semilogy(k, 100 * H["gap"].clip(1e-5), "k^--", ms=4, lw=0.9, label="相对间隙"); a2.set_ylabel("相对间隙/%")
    ax.set_xlabel("迭代次数"); ax.set_ylabel("能源侧年化成本/万元"); ax.set_xticks(k)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="center right")
    save(fig, "图14_CCG收敛", H)


# ------------------------------------------------------------------ 图15 置信水平与样本量（相对最小 θ1 方案的成本增量）
def fig_alpha():
    import acceptance as AC
    A = AC.sens(SITE, "alpha"); Mx = AC.sens(SITE, "M")
    if A is None or len(A) < 3: return
    fig, axs = plt.subplots(2, 1, figsize=(W, 8.0 * CM)); rows = []
    for ax, T, nm, tag in ((axs[0], A, "α1", "(a)"), (axs[1], Mx, "M", "(b)")):
        if T is None or len(T) < 2: ax.set_visible(False); continue
        T = T.sort_values("θ1"); base = T["总成本"].min()
        ax.plot(T["θ1"], T["总成本"] - base, "o-", color="#E64B35", label="交叉评估")
        ax.plot(T["θ1"], T["总成本_单独"] - base, "s", mfc="white", color="gray", label="各点单独求解")
        for _, r in T.iterrows(): ax.annotate(f"{nm}={r['取值']:g}", (r["θ1"], r["总成本"] - base), textcoords="offset points", xytext=(3, 4), fontsize=7)
        ax.set_xlabel("1-范数模糊集半径θ1"); ax.set_ylabel("年化总成本增量/万元"); ax.set_title(tag, fontsize=9, loc="left", pad=2); ax.legend(loc="upper left")
        for _, r in T.iterrows(): rows.append(dict(参数=nm, 取值=r["取值"], θ1=r["θ1"], 总成本=r["总成本"], 增量=r["总成本"] - base, 单独求解增量=r["总成本_单独"] - base))
    fig.tight_layout(h_pad=0.4); save(fig, "图15_置信水平与样本量", pd.DataFrame(rows))


# ------------------------------------------------------------------ 图16 算力冗余率（相对 R_N = 0.2 基准）
def fig_rn():
    rows = []
    for f in [f"{RES}/{SITE}_sens_base.pkl"] + sorted(glob.glob(f"{RES}/{SITE}_sens_RN_*.pkl")):
        if not os.path.exists(f): continue
        r = pickle.load(open(f, "rb")); v = r["params_over"].get("R_N", 0.2)
        e = AN.table_e(SITE, tags={"S5": f}).iloc[0]
        rows.append(dict(R_N=v, N=r["x"]["N"], 能源侧=r["obj"] / 1e4, MC=r["x"]["MC"] / 1e3, TES=r["x"]["TES"] / 1e3, HP=r["x"]["HP"] / 1e3,
                         PUE=e["名义PUE"], 余热=e["余热回收/MWh"] / 1e3, 碳=e["碳排放/t"]))
    if len(rows) < 3: return
    T = pd.DataFrame(rows).sort_values("R_N")
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.6 * CM), sharex=True)
    axs[0].plot(T["R_N"] * 100, T["能源侧"], "o-", color="#3C5488"); axs[0].set_ylabel("能源侧年化成本/万元"); axs[0].set_title("(a)", fontsize=9, loc="left", pad=2)
    axs[1].plot(T["R_N"] * 100, T["MC"], "s-", color="#3C5488", label="机械制冷/MW"); axs[1].plot(T["R_N"] * 100, T["余热"], "^-", color="#E64B35", label="年余热回收/GWh")
    axs[1].set_xlabel("算力冗余率$r_N$/%"); axs[1].set_ylabel("容量或电量"); axs[1].legend(loc="upper left"); axs[1].set_title("(b)", fontsize=9, loc="left", pad=2)
    fig.tight_layout(h_pad=0.3); save(fig, "图16_算力冗余率", T)


# ------------------------------------------------------------------ 液冷热惯性与蓄冷的替代关系（S4，改变 C_w）
def fig_cw():
    rows = []
    for f in sorted(glob.glob(f"{RES}/{SITE}_sensS4_Cw_*.pkl")):
        r = pickle.load(open(f, "rb")); cw = r["params_over"]["C_W"]
        rows.append(dict(C_w=cw, 满载温升=6.0 * 1150.0 / cw, 蓄冷=r["x"]["TES"] / 1e3, 机械制冷=r["x"]["MC"] / 1e3, 能源侧=r["obj"] / 1e4))
    if len(rows) < 3: return
    T = pd.DataFrame(rows).sort_values("C_w")
    fig, ax = plt.subplots(figsize=(W, 5.4 * CM)); x = T["C_w"]
    ax.plot(x, T["蓄冷"], "o-", color="#3C5488", label="蓄冷容量/MWh")
    ax.plot(x, T["机械制冷"] * 10, "s--", color="#E64B35", label="机械制冷容量/(0.1 MW)")
    ax.set_xlabel("液冷环路等效热容$C_w$/(kWh/K)"); ax.set_ylabel("容量"); ax.axvline(1150, color="gray", lw=0.6, ls=":")
    ax.text(1150, ax.get_ylim()[1] * 0.95, " 基准值", fontsize=8, va="top", color="gray")
    a2 = ax.twinx(); a2.plot(x, T["能源侧"], "^:", color="#00A087", label="能源侧年化成本"); a2.set_ylabel("能源侧年化成本/万元")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, "图16b_热惯性与蓄冷替代", T)


# ------------------------------------------------------------------ 图17 芯片温度上限、图18 热价
def _sens(group, key, base_v):
    rows = []
    for f in [f"{RES}/{SITE}_sens_base.pkl"] + sorted(glob.glob(f"{RES}/{SITE}_sens_{group}_*.pkl")):
        if not os.path.exists(f): continue
        r = pickle.load(open(f, "rb")); v = r["params_over"].get(key, base_v)
        rows.append(dict(取值=v, 能源侧=r["obj"] / 1e4, **{k: r["x"][k] for k in ("MC", "FC", "TES", "HP")}, f=f))
    return pd.DataFrame(rows).sort_values("取值") if len(rows) >= 3 else None


def fig_chip():
    T = _sens("chip", "T_CHIP_MAX", 85.0)
    if T is None: return
    b = T[T["取值"] == 85.0].iloc[0]
    fig, ax = plt.subplots(figsize=(W, 5.2 * CM))
    ax.plot(T["取值"], T["MC"] / 1e3, "s-", color="#3C5488", label="机械制冷容量")
    a2 = ax.twinx(); a2.plot(T["取值"], 100 * (T["能源侧"] / b["能源侧"] - 1), "o--", color="#E64B35", label="能源侧成本变化率")
    a2.axhline(0, color="#E64B35", lw=0.5, ls=":"); a2.set_ylabel("相对85 ℃变化率/%")
    ax.set_xlabel("芯片温度上限/℃"); ax.set_ylabel("机械制冷容量/MW")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "图17_芯片温度上限", T.drop(columns="f"))


def fig_heat():
    T = _sens("heat", "C_HEAT", 0.20)
    if T is None: return
    rec = []
    for f in T["f"]: rec.append(AN.table_e(SITE, tags={"S5": f}).iloc[0]["余热回收/MWh"] / 1e3)
    T["余热"] = rec
    fig, ax = plt.subplots(figsize=(W, 5.2 * CM))
    ax.plot(T["取值"], T["HP"] / 1e3, "o-", color="#E64B35", label="热泵容量")
    a2 = ax.twinx(); a2.plot(T["取值"], T["余热"], "s--", color="#3C5488", label="年余热回收"); a2.set_ylabel("年余热回收/GWh")
    ax.set_xlabel("热价/(元/kWh$_{\\rm th}$)"); ax.set_ylabel("热泵容量/MW")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "图18_热价", T.drop(columns="f"))


# ------------------------------------------------------------------ 图19 气候区对比（相对中卫的变化率）
def fig_climate():
    rows = []
    for s, nm in (("zhongwei", "中卫"), ("guian", "贵安"), ("shaoguan", "韶关")):
        if not os.path.exists(f"{RES}/{s}_DRO_S5.pkl"): continue
        e = AN.table_e(s, tags={"S5": "DRO_S5"}).iloc[0]
        rows.append(dict(站点=nm, 机械制冷=e["K_MC/MW"], 自然冷却冷量占比=e["自然冷却冷量占比/%"], PUE=e["名义PUE"], 余热=e["余热回收/MWh"] / 1e3,
                         能源侧=e["能源侧年化成本/万元"]))
    if len(rows) < 3: return
    T = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 2, figsize=(W, 7.0 * CM)); cs = ["#3C5488", "#00A087", "#E64B35"]
    for ax, (k, yl), tag in zip(axs.ravel(), [("机械制冷", "机械制冷/MW"), ("自然冷却冷量占比", "自然冷却冷量占比/%"), ("PUE", "PUE"), ("余热", "年余热回收/GWh")], "abcd"):
        ax.bar(range(3), T[k], 0.6, color=cs, edgecolor="black", linewidth=0.4); ax.set_xticks(range(3)); ax.set_xticklabels(T["站点"])
        ax.set_ylabel(yl); ax.set_title(f"({tag})", fontsize=9, loc="left", pad=2)
        if k == "PUE": lo, hi = T[k].min(), T[k].max(); ax.set_ylim(lo - 0.6 * (hi - lo) - 0.002, hi + 0.5 * (hi - lo) + 0.002)
        if k == "自然冷却冷量占比": lo, hi = T[k].min(), T[k].max(); ax.set_ylim(max(0, lo - 0.8 * (hi - lo) - 1), min(100, hi + 0.5 * (hi - lo) + 1))
        for i, x in enumerate(T[k]): ax.text(i, x, f"{x:.3f}" if k == "PUE" else f"{x:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(h_pad=0.4, w_pad=0.6); save(fig, "图19_气候区对比", T)


if __name__ == "__main__":
    E = tab()
    for f in (lambda: fig_waterfall(E), lambda: fig_capacity(E), lambda: fig_kpi(E), fig_gears, fig_thermal, fig_shift, fig_winter,
              fig_oos, fig_ccg, fig_alpha, fig_rn, fig_cw, fig_chip, fig_heat, fig_climate):
        try: f()
        except Exception as e:
            import traceback; print("出错：", repr(e)); traceback.print_exc(limit=2)
