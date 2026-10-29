"""
图组 B（结果图重设计）
规范: 单栏 8.5 cm; 600 dpi PNG; 中文宋体 + 英文/数字 Times New Roman; 坐标标题 10 pt, 刻度/图例 9 pt;
      主线宽 1.1 pt, Marker 4.8 pt; 白底; 图例无边框。
原则: ① 不改动原始数据; ② 每图 ≤3 组数据, 优先 Baseline 与单一场景的两两对比; ③ 差异小的量改画
      "与 Baseline 的差值"或"相对变化率", 关键时段单独放大; ④ 柱状图纵轴一律从 0 起; 折线/差值图保留零线,
      不截断坐标轴, 避免误导。
运行: python plot_b.py   (环境变量 RESDIR 可指定结果目录)
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import config as C
from plot_style import style_b, COL, MK, W_SINGLE

RES = Path(os.environ.get("RESDIR", C.RES))
OUT = C.FIG / "style_b"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.png"): f.unlink()
TS = pd.read_csv(RES / "timeseries.csv"); SUM = pd.read_csv(RES / "summary.csv").set_index("模型")
TS["冷源电耗"] = TS["自然冷却电耗"] + TS["机械制冷电耗"]
style_b()
H = 6.0 / 2.54
SHORT = {"Baseline": "Baseline", "M1-调频": "M1\n调频", "M2-热耦合": "M2\n热耦合", "M3-SP": "M3\nSP", "M4-DRO": "M4\nDRO",
         "M5-RO": "M5\nRO", "M6-DRO+极端": "M6\nDRO+极端"}
LAB = {"Baseline": "Baseline", "M1-调频": "M1 调频", "M2-热耦合": "M2 热耦合", "M3-SP": "M3 SP",
       "M4-DRO": "M4 DRO(本文)", "M5-RO": "M5 RO", "M6-DRO+极端": "M6 DRO+极端"}
BASE = COL["Baseline"]; PROP = COL["M4-DRO"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=600, facecolor="white"); plt.close(fig)


def fig_(h=H, rows=1, hr=None):
    if rows == 1: return plt.subplots(figsize=(W_SINGLE, h), layout="constrained")
    return plt.subplots(rows, 1, figsize=(W_SINGLE, h), layout="constrained", sharex=True,
                        gridspec_kw=dict(height_ratios=hr or [1] * rows))


def leg(fig, ax, ncol=3, handles=None):
    h, l = handles if handles else ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=ncol, columnspacing=0.9, handlelength=1.7, handletextpad=0.35)


def day(m, d, s): return TS[(TS.模型 == m) & (TS.d == d) & (TS.s == s)].sort_values("t")


def hot_day(d=1):
    g = TS[(TS.模型 == "Baseline") & (TS.d == d)].groupby("s")["环境温度"].max(); return int(g.idxmax())


def taxis(ax, lo=0, hi=24, step=4):
    ax.set_xlim(lo, hi); ax.set_xticks(range(lo, hi + 1, step))


D_, S_ = 1, hot_day(1)
X = np.arange(C.T) + 0.5


# ---------------- B1 冷源容量配置: Baseline vs M2 vs M4 (柱状, 0 起) ----------------
def b1():
    ms = ["Baseline", "M2-热耦合", "M4-DRO"]; w = 0.36; x = np.arange(len(ms))
    fig, ax = fig_()
    fc = [SUM.loc[m, "自然冷却kWth"] for m in ms]; mc = [max(SUM.loc[m, "机械制冷kWth"], 0) for m in ms]
    b1_ = ax.bar(x - w / 2, fc, w, color="#91D1C2", edgecolor="#00A087", lw=0.6, label="自然冷却 $K^{\\mathrm{FC}}$")
    b2_ = ax.bar(x + w / 2, mc, w, color="#F7B6A8", edgecolor="#E64B35", lw=0.6, label="机械制冷 $K^{\\mathrm{MC}}$")
    for bars in (b1_, b2_):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 25, f"{r.get_height():.0f}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in ms]); ax.set_ylabel("冷源容量/kW$_{\\mathregular{th}}$")
    ax.set_ylim(0, max(fc + mc) * 1.18); leg(fig, ax, 2)
    save(fig, "B1_冷源容量配置_Baseline对比.png")


# ---------------- B2 高温日冷源电功率: Baseline vs M4, 全天 + 差值 ----------------
def b2():
    b, p = day("Baseline", D_, S_)["冷源电耗"].values, day("M4-DRO", D_, S_)["冷源电耗"].values
    fig, (ax, bx) = fig_(8.4 / 2.54, rows=2, hr=[1.35, 1])
    ax.step(np.arange(25), np.r_[b, b[-1]], where="post", color=BASE, label=LAB["Baseline"])
    ax.step(np.arange(25), np.r_[p, p[-1]], where="post", color=PROP, label=LAB["M4-DRO"])
    ax.fill_between(np.arange(25), np.r_[b, b[-1]], np.r_[p, p[-1]], step="post", color=PROP, alpha=0.12, lw=0)
    ax.set_ylabel("冷源电功率/kW"); ax.set_ylim(0, max(b.max(), p.max()) * 1.15)
    dv = p - b
    bx.bar(X, dv, 0.8, color=np.where(dv < 0, PROP, BASE), lw=0)
    bx.axhline(0, color="0.5", lw=0.6); bx.set_ylabel("差值/kW"); bx.set_xlabel("时段/$t$"); taxis(bx)
    rel = dv.sum() / b.sum() * 100
    bx.set_ylim(dv.min() * 1.32, -dv.min() * 0.06)  # 下方留出标注空间, 0 线仍可见
    bx.text(0.98, 0.05, f"全天电量 {dv.sum():+.0f} kWh（{rel:+.1f}%）", transform=bx.transAxes, ha="right", fontsize=8)
    leg(fig, ax, 2)
    save(fig, "B2_高温日冷源电功率_Baseline对比.png")


# ---------------- B3 高温日液冷温度: Baseline / M2 / M4 + 模式安全上限 ----------------
def b3():
    fig, ax = fig_(6.6 / 2.54)
    for (lo, hi), c in zip(C.ZONES, ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#FDD0A2", "#FCBBA1"]):
        ax.axhspan(lo, hi, color=c, alpha=0.65, lw=0, zorder=0)
    for k, ls in zip(C.MODES, ["--", "-.", ":"]):
        lim = C.T_CHIP_MAX - C.R_TH * C.MODES[k][1]; ax.axhline(lim, color="k", ls=ls, lw=0.7)
        ax.text(23.7, lim + 0.2, k.split("_")[1] + "安全上限", fontsize=7.5, ha="right", va="bottom")
    for m in ["Baseline", "M2-热耦合", "M4-DRO"]:
        y = day(m, D_, S_)["液冷温度"].values; y = np.r_[y, y[0]]
        ax.plot(range(25), y, color=COL[m], marker=MK[m], markevery=3, label=LAB[m])
    taxis(ax); ax.set_xlabel("时段/$t$"); ax.set_ylabel("液冷等效温度 $T^w$/℃")
    ax.set_ylim(C.ZONES[0][0] - 1, C.ZONES[-1][1] + 2.2); leg(fig, ax, 3)
    save(fig, "B3_高温日液冷温度_安全温区.png")


# ---------------- B4 高温日 IT 功率: Baseline vs M1, 绝对值 + 相对变化率 ----------------
def b4():
    b, p = day("Baseline", D_, S_)["IT功率"].values, day("M1-调频", D_, S_)["IT功率"].values
    fig, (ax, bx) = fig_(8.4 / 2.54, rows=2, hr=[1.35, 1])
    ax.plot(X, b / 1e3, color=BASE, marker=MK["Baseline"], markevery=2, label=LAB["Baseline"])
    ax.plot(X, p / 1e3, color=COL["M1-调频"], marker=MK["M1-调频"], markevery=2, label=LAB["M1-调频"])
    ax.set_ylabel("IT功率/MW"); ax.set_ylim(0, max(b.max(), p.max()) / 1e3 * 1.15)
    r = (p / b - 1) * 100
    bx.bar(X, r, 0.8, color=COL["M1-调频"], lw=0); bx.axhline(0, color="0.5", lw=0.6)
    bx.set_ylabel("相对变化率/%"); bx.set_xlabel("时段/$t$"); taxis(bx)
    bx.set_ylim(min(r.min() * 1.25, -1), max(r.max() * 1.25, 0.5))
    leg(fig, ax, 2)
    save(fig, "B4_高温日IT功率_GPU调节效果.png")


# ---------------- B5 Baseline vs M4: 年度指标相对变化率 (单模型, 0 线) ----------------
def b5():
    S = SUM.copy(); S["冷源年化投资"] = S["年化_FC"] + S["年化_MC"]
    ind = [("机械制冷kWth", "冷水机组容量"), ("冷源年化投资", "冷源年化投资"), ("冷却电量MWh", "冷源电量"),
           ("能源相关费用_经验", "能源相关年费用"), ("期望电费_经验", "期望电费"), ("IT电量MWh", "IT电量"), ("PUE", "PUE")]
    rel = np.array([(S.loc["M4-DRO", k] / S.loc["Baseline", k] - 1) * 100 for k, _ in ind])
    fig, ax = fig_(6.4 / 2.54)
    y = np.arange(len(ind))
    ax.barh(y, rel, 0.62, color=PROP, lw=0)
    for i, v in enumerate(rel):
        ax.text(v - 1.5 if v < 0 else v + 1.5, i, f"{v:+.1f}%", va="center", ha="right" if v < 0 else "left", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in ind]); ax.invert_yaxis(); ax.axvline(0, color="0.4", lw=0.7)
    ax.set_xlim(min(rel.min() * 1.3, -5), 8); ax.set_xlabel("M4 相对 Baseline 变化率/%")
    save(fig, "B5_M4相对Baseline年度指标变化率.png")


# ---------------- B6 消融: 能源相关年费用相对 Baseline 的差值 (单序列) ----------------
def b6():
    ms = ["M1-调频", "M2-热耦合", "M3-SP", "M4-DRO", "M6-DRO+极端"]
    dv = np.array([(SUM.loc[m, "能源相关费用_经验"] - SUM.loc["Baseline", "能源相关费用_经验"]) / 1e4 for m in ms])
    gap = np.array([SUM.loc[m, "间隙"] / 1e4 for m in ms])
    fig, ax = fig_()
    x = np.arange(len(ms))
    ax.bar(x, dv, 0.6, color=[COL[m] for m in ms], lw=0)
    ax.errorbar(x, dv, yerr=gap, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2.5)
    for i, v in enumerate(dv): ax.text(i, v - gap[i] - 1.0, f"{v:.1f}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.7); ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ms])
    ax.set_ylabel("能源相关年费用差值/万元"); ax.set_ylim((dv - gap).min() * 1.3, max(5, gap.max() * 1.5))
    ax.text(0.99, 0.97, "相对 Baseline；误差线为 MILP 绝对间隙", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    save(fig, "B6_消融对比_能源相关费用差值.png")


# ---------------- B7 样本外逐日运行费用节省 (Baseline − M4), 分季节箱线图 ----------------
def b7():
    b = pd.read_csv(RES / "oos_Baseline.csv").set_index(["d", "i"])["日运行费用"]
    p = pd.read_csv(RES / "oos_M4-DRO.csv").set_index(["d", "i"])["日运行费用"]
    sv = (b - p).dropna().rename("节省").reset_index()
    fig, ax = fig_()
    data = [sv[sv.d == d]["节省"].values for d in range(4)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color="k", lw=1.0), boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8), flierprops=dict(marker="o", ms=3, mfc="none", mec="0.4", mew=0.6))
    for patch in bp["boxes"]: patch.set_facecolor("#F7B6A8"); patch.set_edgecolor(PROP)
    for i, v in enumerate(data): ax.plot(i + 1, v.mean(), marker="^", color=PROP, ms=4.8, zorder=3)
    ax.axhline(0, color="0.4", lw=0.7); ax.set_xticks(range(1, 5)); ax.set_xticklabels(["春季", "夏季", "秋季", "冬季"])
    lo = min(0, min(v.min() for v in data)); hi = max(v.max() for v in data)
    ax.set_ylabel("逐日运行费用节省/元"); ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.text(0.99, 0.97, "▲ 均值；同一测试日配对比较", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    save(fig, "B7_样本外逐日节省_Baseline减M4.png")


# ---------------- B8 样本外可靠性: 实时任务缺额 (单序列, 0 起) ----------------
def b8():
    ms = ["Baseline", "M3-SP", "M4-DRO", "M6-DRO+极端"]
    v = [SUM.loc[m, "实时缺额GPUh"] for m in ms]; dd = [SUM.loc[m, "缺额日比例"] * 100 for m in ms]
    fig, ax = fig_()
    x = np.arange(len(ms)); bars = ax.bar(x, v, 0.6, color=[COL[m] for m in ms], lw=0)
    for i, r in enumerate(bars):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + max(v) * 0.02, f"{v[i]:.0f}\n（{dd[i]:.1f}%日）",
                ha="center", va="bottom", fontsize=7.5, linespacing=1.1)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ms]); ax.set_ylim(0, max(v) * 1.3)
    ax.set_ylabel("测试日实时任务缺额/(GPU·h)")
    save(fig, "B8_样本外实时任务缺额.png")


# ---------------- B9 夏季情景: 经验概率 vs 最坏分布 (M4) ----------------
def b9():
    P = pd.read_csv(RES / "probabilities.csv"); P = P[(P.模型 == "M4-DRO") & (P.类别 == "夏季")]
    g = TS[(TS.模型 == "Baseline") & (TS.d == 1)].groupby("s")
    tmax, bday = g["环境温度"].max(), g["延期到达"].sum()
    fig, ax = fig_(6.4 / 2.54)
    x = np.arange(len(P)); w = 0.38
    ax.bar(x - w / 2, P["经验概率"], w, color="#B8C3DE", edgecolor="#3C5488", lw=0.6, label="经验概率 $\\hat{p}_{ds}$")
    ax.bar(x + w / 2, P["最坏概率"], w, color="#F7B6A8", edgecolor=PROP, lw=0.6, label="最坏分布 $p^{*}_{ds}$")
    ax.set_xticks(x); ax.set_xticklabels([f"{tmax[s]:.0f}℃\n{bday[s] / 1e3:.1f}" for s in P["情景"]])
    ax.set_xlabel("夏季情景：日最高气温 /\n延期任务日到达量(10$^{\\mathregular{3}}$ GPU·h)")
    ax.set_ylabel("情景概率"); ax.set_ylim(0, max(P["经验概率"].max(), P["最坏概率"].max()) * 1.2)
    leg(fig, ax, 2)
    save(fig, "B9_夏季经验概率与最坏分布.png")


# ---------------- B10 模糊集缩放 κ: (a) 总费用差值 (b) 容量 ----------------
def b10():
    SE = pd.read_csv(RES / "sensitivity.csv"); base = SUM.loc["M4-DRO"]
    k = SE[SE.参数 == "kappa"][["取值", "总费用_最坏", "总费用_经验", "光伏kWp", "机械制冷kWth"]]
    k = pd.concat([k, pd.DataFrame([dict(取值=1.0, 总费用_最坏=base["总费用_最坏"], 总费用_经验=base["总费用_经验"],
                                          光伏kWp=base["光伏kWp"], 机械制冷kWth=base["机械制冷kWth"])])]).sort_values("取值")
    fig, (ax, bx) = fig_(8.0 / 2.54, rows=2)
    ax.plot(k.取值, (k.总费用_最坏 - base["总费用_最坏"]) / 1e4, color=PROP, marker="^", label="最坏分布期望")
    ax.plot(k.取值, (k.总费用_经验 - base["总费用_经验"]) / 1e4, color="#3C5488", marker="o", label="经验分布期望")
    ax.axhline(0, color="0.5", lw=0.6); ax.set_ylabel("总费用差值/万元")
    bx.plot(k.取值, k.光伏kWp, color="#00A087", marker="s", label="光伏 $K^{\\mathrm{PV}}$")
    bx.plot(k.取值, k.机械制冷kWth.clip(lower=0), color="#B09C85", marker="D", label="冷水机组 $K^{\\mathrm{MC}}$")
    bx.set_ylabel("容量/kW"); bx.set_xlabel("模糊集缩放系数 $\\kappa$"); bx.set_ylim(0, None)
    bx.legend(loc="center left", frameon=False)
    leg(fig, ax, 2)
    save(fig, "B10_模糊集缩放敏感性.png")


# ---------------- B11 关键参数: 能源相关年费用变化率 (单序列, 标注结构变化) ----------------
def b11():
    SE = pd.read_csv(RES / "sensitivity.csv"); base = SUM.loc["M4-DRO"]
    rows = SE[(SE.参数 != "kappa") & (SE.模型 == "M4-DRO")].copy()
    lab = {"FLOAT": "峰谷浮动", "R_TH": "热阻倍数", "BETA_LIQ": "液冷比例", "C_W": "环路热容"}
    rel = (rows["能源相关费用_经验"] / base["能源相关费用_经验"] - 1).values * 100
    fig, ax = fig_()
    y = np.arange(len(rows))
    ax.barh(y, rel, 0.6, color=np.where(rel > 0, PROP, "#3C5488"), lw=0)
    ax.set_yticks(y); ax.set_yticklabels([f"{lab[r.参数]}={r.取值:g}" for r in rows.itertuples()]); ax.invert_yaxis()
    ax.axvline(0, color="0.4", lw=0.7); ax.set_xlabel("能源相关年费用变化率/%")
    lim = max(abs(rel).max() * 1.2, 1); ax.set_xlim(-lim, lim)
    for i, r in enumerate(rows.itertuples()):
        note = []
        if r.机械制冷kWth > 1: note.append(f"冷机{r.机械制冷kWth:.0f}kW")
        if r.储能kWh > 1: note.append(f"储能{r.储能kWh / 1e3:.1f}MWh")
        if r.光伏kWp < 1: note.append("光伏0")
        if r.GPU台数 != base["GPU台数"]: note.append(f"GPU{r.GPU台数:.0f}")
        if note:
            ax.text(-0.15 if rel[i] > 0 else 0.15, i, "，".join(note), va="center", ha="right" if rel[i] > 0 else "left",
                    fontsize=7.5)
    save(fig, "B11_关键参数敏感性.png")


if __name__ == "__main__":
    for f in [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]:
        try: f(); print("ok", f.__name__)
        except Exception as e: print("跳过", f.__name__, repr(e))
