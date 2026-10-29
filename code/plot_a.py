"""
图组 A (时序对比, 23 pt Times New Roman, 四边封闭, 横轴 时段/t 从 0 开始)
读取 results/timeseries.csv, results/summary.csv; 输出 figures/style_a/*.png (600 dpi)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import config as C
from plot_style import style_a, frame_a, time_axis, free_ylim, stairs, save, COL, MK, LAB

OUT = C.FIG / "style_a"; OUT.mkdir(parents=True, exist_ok=True)
TS = pd.read_csv(C.RES / "timeseries.csv"); SUM = pd.read_csv(C.RES / "summary.csv")
style_a()
FS = (11.5, 7.2)


def pick(d=1, how="hot"):
    """夏季(d=1)中最高气温情景 = 冷却最紧张的代表日。"""
    g = TS[(TS.模型 == "Baseline") & (TS.d == d)].groupby("s")["环境温度"].max()
    return int(g.idxmax() if how == "hot" else g.idxmin())


def day(model, d, s):
    return TS[(TS.模型 == model) & (TS.d == d) & (TS.s == s)].sort_values("t")


def legend(ax, ncol=2, **kw):
    ax.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.01), handlelength=1.6, columnspacing=1.0,
              borderaxespad=0.1, **kw)   # 图例置于坐标区上方, 避免遮挡曲线


def fig_series(key, models, ylabel, name, d, s, state=False, scale=1.0, extra=None):
    fig, ax = plt.subplots(figsize=FS); data = []
    for m in models:
        y = day(m, d, s)[key].values * scale; data.append(y)
        if state:   # 状态量 t=0..T-1 为时段起点值, 末端用初值闭合 (式2/5/11 初末条件)
            yy = np.r_[y, y[0]]; ax.plot(np.arange(len(yy)), yy, color=COL[m], marker=MK[m], ms=7, label=LAB[m])
        else:
            stairs(ax, y, color=COL[m], label=LAB[m])
    if extra: extra(ax)
    time_axis(ax); ax.set_ylabel(ylabel); free_ylim(ax, *data); frame_a(ax); legend(ax)
    save(fig, OUT / f"{name}.png")


def main():
    d, s = 1, pick(1)
    print("夏季高温代表日: s =", s)
    base4 = ["Baseline", "M1-调频", "M3-SP", "M4-DRO"]
    fig_series("IT功率", base4, "IT功率/MW", "A1_IT功率对比", d, s, scale=1e-3)
    def zones(ax):
        for (lo, hi), c in zip(C.ZONES, ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#FDD0A2", "#FCBBA1"]):
            ax.axhspan(lo, hi, color=c, alpha=0.55, lw=0, zorder=0)
        ax.axhline(C.T_CHIP_MAX - C.R_TH * C.TDP, color="k", ls="--", lw=1.6)
        ax.text(23.6, C.T_CHIP_MAX - C.R_TH * C.TDP - 0.5, "标称模式安全上限", fontsize=23, ha="right", va="top")
    fig_series("液冷温度", ["Baseline", "M2-热耦合", "M3-SP", "M4-DRO"], "液冷等效温度/℃", "A2_液冷温度对比", d, s,
               state=True, extra=zones)
    TS["冷却电耗"] = TS["自然冷却电耗"] + TS["机械制冷电耗"]
    fig_series("冷却电耗", ["Baseline", "M2-热耦合", "M3-SP", "M4-DRO"], "冷源电功率/kW", "A3_冷却电功率对比", d, s)
    fig_series("机械制冷散热", ["Baseline", "M2-热耦合", "M3-SP", "M4-DRO"], "机械制冷散热功率/kW", "A4_机械制冷散热对比", d, s)
    fig_series("延期完成", ["Baseline", "M1-调频", "M3-SP", "M4-DRO"], "可延期任务处理量/(GPU·h)", "A5_延期任务调度对比", d, s)
    # 购电功率 + 分时电价 (右轴)
    fig, ax = plt.subplots(figsize=FS); data = []
    for m in ["Baseline", "M3-SP", "M4-DRO"]:
        y = day(m, d, s)["购电"].values * 1e-3; data.append(y); stairs(ax, y, color=COL[m], label=LAB[m])
    time_axis(ax); ax.set_ylabel("购电功率/MW"); free_ylim(ax, *data); frame_a(ax, right_ticks=False)
    ax2 = ax.twinx(); pr = day("Baseline", d, s)["电价"].values
    stairs(ax2, pr, color="#B09C85", ls=":", lw=2.4, label="分时电价")
    ax2.set_ylabel("电价/(元·kWh$^{\mathregular{-1}}$)"); free_ylim(ax2, pr, pad=0.15); frame_a(ax2)
    ax2.tick_params(left=False, top=False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, columnspacing=1.0,
              handlelength=1.6)
    save(fig, OUT / "A6_购电与电价.png")
    # 本文方法 GPU 运行模式分配 (堆叠)
    g = day("M4-DRO", d, s); cols = [c for c in g.columns if c.startswith("m_")]
    fig, ax = plt.subplots(figsize=FS); bottom = np.zeros(C.T)
    pal = ["#3C5488", "#8491B4", "#00A087", "#91D1C2", "#E64B35"]
    nice = {"R|k1_标称": "实时-标称", "R|k2_浅锁频": "实时-浅锁频", "B|k1_标称": "延期-标称",
            "B|k2_浅锁频": "延期-浅锁频", "B|k3_深锁频": "延期-深锁频"}
    for c, col in zip(cols, pal):
        v = g[c].values
        if v.max() < 1e-6: continue          # 未使用的模式不进入图例
        ax.bar(np.arange(C.T) + 0.5, v, 0.82, bottom=bottom, color=col, label=nice[c[2:]],
                                 edgecolor="white", lw=0.4); bottom += v
    N = SUM.set_index("模型").loc["M4-DRO", "GPU台数"]
    ax.axhline(N, color="k", ls="--", lw=1.6, label="配置GPU数$N$")
    time_axis(ax); ax.set_ylabel("GPU分配数/台"); free_ylim(ax, bottom, [N], lo=0, pad=0.10); frame_a(ax)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, columnspacing=0.8, handlelength=1.2)
    ax.set_ylim(0, N * 1.12)
    save(fig, OUT / "A7_GPU模式分配_M4.png")
    # 光伏可用与利用
    fig, ax = plt.subplots(figsize=FS); data = []
    for m in ["Baseline", "M4-DRO"]:
        g = day(m, d, s)
        stairs(ax, g["光伏可用"].values, color=COL[m], ls="--", label=LAB[m] + " 可用")
        stairs(ax, g["光伏利用"].values, color=COL[m], label=LAB[m] + " 利用"); data += [g["光伏可用"].values]
    time_axis(ax); ax.set_ylabel("光伏功率/kW"); free_ylim(ax, *data, lo=0); frame_a(ax); legend(ax)
    save(fig, OUT / "A8_光伏出力与消纳.png")
    print("图组A完成:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
