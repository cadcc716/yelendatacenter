"""
图组 B (重设计: 突出与 Baseline 的有效差异; 单栏 8.5 cm, 600 dpi; 中文宋体/英文 Times New Roman;
        坐标标题 10 pt, 刻度/图例 9 pt, 线宽 1.1 pt, Marker 4.8 pt, 图例无边框)
原则: 不修改原始数据; 用"与基准差值/相对变化率/局部放大"表达差异; 零线与放大框明确标注, 避免误导。
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import config as C
from plot_style import style_b, COL, MK, W_SINGLE, stairs

LAB = {"Baseline": "Baseline", "M1-调频": "M1 调频", "M2-热耦合": "M2 热耦合", "M3-SP": "M3 SP",
       "M4-DRO": "M4 DRO(本文)", "M5-RO": "M5 RO", "M6-DRO+极端": "M6 DRO+极端"}


def save(fig, path):
    """严格按 8.5 cm 画布输出 (不使用 tight 裁切, 保证单栏宽度), 600 dpi。"""
    fig.savefig(path, dpi=600, facecolor="white"); plt.close(fig)


def leg(fig, ax, ncol=3, handles=None):
    h, l = handles if handles else ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=ncol, columnspacing=0.8, handlelength=1.6, handletextpad=0.3)

OUT = C.FIG / "style_b"; OUT.mkdir(parents=True, exist_ok=True)
TS = pd.read_csv(C.RES / "timeseries.csv"); SUM = pd.read_csv(C.RES / "summary.csv").set_index("模型")
TS["冷却电耗"] = TS["自然冷却电耗"] + TS["机械制冷电耗"]
style_b()
H = 6.0 / 2.54
MODELS = ["M1-调频", "M2-热耦合", "M3-SP", "M4-DRO", "M5-RO", "M6-DRO+极端"]


def day(m, d, s): return TS[(TS.模型 == m) & (TS.d == d) & (TS.s == s)].sort_values("t")


def hot_day(d=1):
    g = TS[(TS.模型 == "Baseline") & (TS.d == d)].groupby("s")["环境温度"].max(); return int(g.idxmax())


def tidy(ax, xlabel="时段/$t$", T=24):
    ax.set_xlim(0, T); ax.set_xticks(range(0, T + 1, 4)); ax.set_xlabel(xlabel)
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)


def b1_delta_cooling(d=1):
    """冷源电功率与 Baseline 的差值 ΔP(t): (a) 全天; (b) 高温时段局部放大 (阴影区对应)。"""
    s = hot_day(d); base = day("Baseline", d, s)
    Ta = base["环境温度"].values; k = int(np.argmax(Ta)); lo, hi = max(k - 4, 0), min(k + 5, C.T)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W_SINGLE, 9.0 / 2.54), layout="constrained")
    x = np.arange(C.T) + 0.5; vals = {}
    for m in ["M2-热耦合", "M3-SP", "M4-DRO"]:
        dv = day(m, d, s)["冷却电耗"].values - base["冷却电耗"].values; vals[m] = dv
        ax.plot(x, dv, color=COL[m], marker=MK[m], markevery=3, label=LAB[m])
        bx.plot(x[lo:hi], dv[lo:hi], color=COL[m], marker=MK[m], label=LAB[m])
    tidy(ax); ax.set_ylabel("$\\Delta P$/kW"); ax.axvspan(lo, hi, color="0.92", zorder=0)
    ax.text(0.01, 0.04, "(a) 全天", transform=ax.transAxes, fontsize=9)
    leg(fig, ax, 3)
    bx.set_xlim(lo, hi); bx.set_xticks(range(lo, hi + 1, 2)); bx.set_xlabel("时段/$t$")
    bx.set_ylabel("$\\Delta P$/kW"); bx.axhline(0, color="0.6", lw=0.6)
    bx.text(0.01, 0.04, f"(b) 高温时段放大 (环境最高 {Ta.max():.1f} ℃)", transform=bx.transAxes, fontsize=9)
    yy = np.concatenate([v[lo:hi] for v in vals.values()]); r = np.ptp(yy) or 1
    bx.set_ylim(yy.min() - 0.25 * r, yy.max() + 0.12 * r)
    save(fig, OUT / "B1_冷源电功率较基准差值.png")


def b2_loop_temp(d=1):
    """液冷等效温度轨迹 + 温区色带 + 各 GPU 模式安全上限。"""
    s = hot_day(d)
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    for (lo, hi), c in zip(C.ZONES, ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#FDD0A2", "#FCBBA1"]):
        ax.axhspan(lo, hi, color=c, alpha=0.6, lw=0, zorder=0)
    for k, ls in zip(C.MODES, ["--", "-.", ":"]):
        lim = C.T_CHIP_MAX - C.R_TH * C.MODES[k][1]; ax.axhline(lim, color="k", ls=ls, lw=0.7)
        ax.text(0.3, lim + 0.25, k.split("_")[1] + "上限", fontsize=7.5, va="bottom")
    for m in ["Baseline", "M2-热耦合", "M3-SP", "M4-DRO"]:
        y = day(m, d, s)["液冷温度"].values; y = np.r_[y, y[0]]
        ax.plot(range(C.T + 1), y, color=COL[m], marker=MK[m], markevery=3, label=LAB[m])
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.set_xlabel("时段/$t$")
    ax.set_ylim(C.ZONES[0][0] - 1, C.ZONES[-1][1] + 2.6); ax.set_ylabel("液冷等效温度 $T^w$/℃")
    leg(fig, ax, 2)
    save(fig, OUT / "B2_液冷温度轨迹与安全上限.png")


def b3_relative_change():
    """年度指标相对 Baseline 的变化率 (%); 每个指标一行, 不同模型用不同符号。"""
    ind = {"冷源年化投资": None, "期望电费_经验": "期望电费", "冷却电量MWh": "冷却电量", "机械制冷容量": None,
           "能源相关费用_经验": "能源相关费用", "PUE": "PUE"}
    S = SUM.copy(); S["冷源年化投资"] = S["年化_FC"] + S["年化_MC"]; S["机械制冷容量"] = S["机械制冷kWth"]
    keys = list(ind)
    fig, ax = plt.subplots(figsize=(W_SINGLE, 6.4 / 2.54), layout="constrained")
    for j, m in enumerate(MODELS):
        rel = [(S.loc[m, k] / S.loc["Baseline", k] - 1) * 100 for k in keys]
        ax.scatter(rel, np.arange(len(keys)) + (j - 2.5) * 0.12, color=COL[m], marker=MK[m], s=22, label=LAB[m], zorder=3)
    ax.axvline(0, color="0.5", lw=0.6); ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([ind[k] or k for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("相对 Baseline 变化率/%")
    leg(fig, ax, 3)
    ax.grid(axis="x", color="0.9", lw=0.5)
    save(fig, OUT / "B3_年度指标相对变化率.png")


def b4_oos():
    """样本外: 测试日与分布偏移 θ 下的年运行费用相对 Baseline 变化率。"""
    cols = ["样本外年运行费用", "偏移θ=0.5年运行费用", "偏移θ=1.0年运行费用"]; xs = [0, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    for m in MODELS:
        y = [(SUM.loc[m, c] / SUM.loc["Baseline", c] - 1) * 100 for c in cols]
        ax.plot(xs, y, color=COL[m], marker=MK[m], label=LAB[m])
    ax.set_xlabel("分布偏移强度 $\\theta$"); ax.set_ylabel("样本外运行费用变化率/%"); ax.set_xticks(xs)
    ax.axhline(0, color="0.6", lw=0.6)
    leg(fig, ax, 3)
    save(fig, OUT / "B4_样本外运行费用_分布偏移.png")


def b5_paired_daily():
    """样本外逐日配对差值 (同一测试日, 模型费用 − Baseline 费用) 的经验分布。"""
    E = {m: pd.read_csv(C.RES / f"oos_{m}.csv") for m in ["Baseline"] + MODELS}
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    b = E["Baseline"].set_index(["d", "i"])["日运行费用"]
    for m in MODELS:
        dv = (E[m].set_index(["d", "i"])["日运行费用"] - b).dropna().sort_values().values / 1e3
        ax.plot(dv, np.linspace(0, 1, len(dv)), color=COL[m], label=LAB[m])
    ax.axvline(0, color="0.6", lw=0.6); ax.set_xlabel("逐日运行费用差值/(10$^3$元)"); ax.set_ylabel("累积概率")
    leg(fig, ax, 3)
    ax.set_ylim(0, 1.02)
    save(fig, OUT / "B5_样本外逐日费用差值CDF.png")


def b8_shortfall():
    """样本外未满足工作量与运行费用的权衡 (点旁直接标注模型, 不用图例)。"""
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    off = {"Baseline": (-4, 6, "right"), "M1-调频": (5, 2, "left"), "M2-热耦合": (5, -10, "left"), "M3-SP": (6, -3, "left"),
           "M4-DRO": (6, -3, "left"), "M5-RO": (6, -3, "left"), "M6-DRO+极端": (6, -3, "left")}
    for m in ["Baseline"] + MODELS:
        x = (SUM.loc[m, "样本外年运行费用"] / SUM.loc["Baseline", "样本外年运行费用"] - 1) * 100
        y = SUM.loc[m, "实时缺额GPUh"]
        ax.scatter(x, y, color=COL[m], marker=MK[m], s=30, zorder=3)
        dx, dy, ha = off[m]
        ax.annotate(LAB[m], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=COL[m])
    ax.set_xlabel("样本外运行费用变化率/%"); ax.set_ylabel("实时任务缺额/(GPU·h)")
    ax.set_xlim(-5.2, 0.9); ax.grid(color="0.92", lw=0.5)
    save(fig, OUT / "B8_样本外费用与缺额权衡.png")


def b6_probabilities():
    P = pd.read_csv(C.RES / "probabilities.csv"); P = P[P.模型 == "M4-DRO"]
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained"); x = np.arange(len(P)); w = 0.38
    ax.bar(x - w / 2, P["经验概率"], w, color="#8491B4", label="经验概率 $\\hat p$")
    ax.bar(x + w / 2, P["最坏概率"], w, color="#E64B35", label="最坏分布 $p^*$")
    for k in range(1, 4): ax.axvline(k * C.N_SCEN - 0.5, color="0.7", lw=0.5)
    ax.set_xticks([k * C.N_SCEN + 2 for k in range(4)]); ax.set_xticklabels(P["类别"].unique())
    ax.set_ylabel("情景概率"); leg(fig, ax, 2)
    save(fig, OUT / "B6_经验与最坏情景概率.png")


def b7_sensitivity():
    S = pd.read_csv(C.RES / "sensitivity.csv")
    base = SUM.loc["M4-DRO"]
    # (a) 模糊集缩放 κ
    k = S[S.参数 == "kappa"][["取值", "总费用_经验", "总费用_最坏", "机械制冷kWth", "光伏kWp"]]
    k = pd.concat([k, pd.DataFrame([dict(取值=1.0, 总费用_经验=base["总费用_经验"], 总费用_最坏=base["总费用_最坏"],
                                          机械制冷kWth=base["机械制冷kWth"], 光伏kWp=base["光伏kWp"])])]).sort_values("取值")
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    ax.plot(k.取值, (k.总费用_最坏 - base["总费用_最坏"]) / 1e4, color="#E64B35", marker="^", label="最坏期望总费用")
    ax.plot(k.取值, (k.总费用_经验 - base["总费用_经验"]) / 1e4, color="#3C5488", marker="o", label="经验期望总费用")
    ax.set_xlabel("模糊集缩放系数 $\\kappa$"); ax.set_ylabel("相对 $\\kappa$=1 的费用差值/万元"); ax.axhline(0, color="0.6", lw=0.6)
    ax2 = ax.twinx(); ax2.plot(k.取值, k.光伏kWp, color="#00A087", marker="s", ls="--", label="光伏容量")
    ax2.set_ylabel("光伏容量/kWp"); ax2.tick_params(direction="in")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    leg(fig, ax, 2, (h1 + h2, l1 + l2))
    save(fig, OUT / "B7a_模糊集缩放敏感性.png")
    # (b) 单参数扰动对能源相关费用与机械制冷容量的影响
    rows = S[(S.参数 != "kappa") & (S.模型 == "M4-DRO")]
    lab = {"FLOAT": "峰谷浮动", "R_TH": "热阻倍数", "BETA_LIQ": "液冷比例", "C_W": "环路热容"}
    fig, ax = plt.subplots(figsize=(W_SINGLE, H), layout="constrained")
    names = [f"{lab[r.参数]}={r.取值:g}" for r in rows.itertuples()]
    rel = (rows["能源相关费用_经验"] / base["能源相关费用_经验"] - 1) * 100
    ax.barh(range(len(rows)), rel, color=["#E64B35" if v > 0 else "#3C5488" for v in rel], height=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlabel("能源相关年费用变化率/%")
    for i, r in enumerate(rows.itertuples()):   # 只标注结构性变化: 冷水机组、储能、光伏归零
        note = []
        if r.机械制冷kWth > 1: note.append(f"冷机 {r.机械制冷kWth:.0f} kW")
        if r.储能kWh > 1: note.append(f"储能 {r.储能kWh / 1e3:.1f} MWh")
        if r.光伏kWp < 1: note.append("光伏 0")
        if r.GPU台数 != base["GPU台数"]: note.append(f"GPU {r.GPU台数:.0f}")
        if note:
            v = rel.iloc[i]
            ax.text(-0.15 if v > 0 else 0.15, i, ", ".join(note), va="center", ha="right" if v > 0 else "left", fontsize=7.5)
    lim = max(abs(rel).max() * 1.15, 1); ax.set_xlim(-lim, lim)
    save(fig, OUT / "B7b_关键参数敏感性.png")


if __name__ == "__main__":
    for f in [b1_delta_cooling, b2_loop_temp, b3_relative_change, b4_oos, b5_paired_daily, b6_probabilities, b7_sensitivity, b8_shortfall]:
        try: f(); print("ok", f.__name__)
        except Exception as e: print("跳过", f.__name__, repr(e))
