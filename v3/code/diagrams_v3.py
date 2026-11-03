"""
v3 论文示意图: 图1 系统结构 / 图2 "频率—功耗—温度—冷却—余热"闭环 / 图3 C&CG 流程 (PDF/SVG + 600 dpi PNG)
线型: 电功率流实线, 冷/热流虚线, 信息约束点划线。字体: 西文/公式 Times New Roman, 中文宋体。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import plot_style as PS

OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(parents=True, exist_ok=True)
PS._fonts(8); plt.rcParams.update({"font.size": 8})
EC = {"电": "#3C5488", "热": "#C0392B", "冷": "#2E86C1", "算": "#00876C", "信": "#6E6E6E"}
FC_ = {"电": "#EDF1F9", "热": "#FBECE9", "冷": "#EAF4FB", "算": "#E6F3EF", "信": "#F4F4F4", "无": "white"}
LS = {"电": "-", "热": "--", "冷": "--", "算": "-", "信": "-."}


def canvas(w, h):
    fig = plt.figure(figsize=(w / 2.54, h / 2.54)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", kind="无", title=None, fs=7.2, ec=None, lw=0.8, r=0.12, tfs=7.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ls=ls, ec=ec or EC.get(kind, "#555"), fc=FC_[kind], zorder=2))
    if title:
        ax.text(x + w / 2, y + h - 0.1, title, ha="center", va="top", fontsize=tfs, fontweight="bold", zorder=3); ty = y + (h - 0.4) / 2
    else:
        ty = y + h / 2
    if text: ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs, linespacing=1.3, zorder=3)


def arrow(ax, pts, kind, label=None, lpos=0.5, loff=(0, 0.1), ha="center", va="bottom", fs=6.8, both=False):
    pts = np.asarray(pts, float); col = EC[kind]; ls = LS[kind]
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=col, lw=1.0, ls=ls, zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1, arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.2", color=col, lw=1.0, ls=ls, shrinkA=0, shrinkB=0))
    if both: ax.annotate("", xy=pts[0], xytext=pts[1], zorder=1, arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.2", color=col, lw=1.0, shrinkA=0, shrinkB=0))
    if label:
        seg = np.diff(pts, axis=0); L = np.hypot(*seg.T); s = lpos * L.sum(); i = min(int(np.searchsorted(np.cumsum(L), s)), len(seg) - 1)
        f = (s - np.r_[0, np.cumsum(L)][i]) / max(L[i], 1e-9); p = pts[i] + f * seg[i]
        ax.text(p[0] + loff[0], p[1] + loff[1], label, ha=ha, va=va, fontsize=fs, color=col, zorder=3)


def save(fig, name):
    for ext in ("pdf", "svg"): fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04); plt.close(fig); print("saved", name)


def fig1():
    W, H = 17.0, 12.2; fig, ax = canvas(W, H)
    for y0, y1, nm, c in ((8.55, 11.9, "算力层", "算"), (4.35, 8.35, "热管理层", "热"), (0.3, 4.15, "能源层", "电")):
        ax.add_patch(FancyBboxPatch((0.25, y0), W - 0.5, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15", lw=0.6, ls=(0, (4, 2)), ec="#AAAAAA", fc="none", zorder=0))
        ax.text(0.45, y1 - 0.15, nm, ha="left", va="top", fontsize=8.5, fontweight="bold", color=EC[c])
    # 算力层
    box(ax, 1.0, 10.15, 3.3, 1.15, "推理请求 $A^{\\mathrm{R}}_{st}$\n档位 $k∈\\{1,2,3\\}$", "算", "实时任务")
    box(ax, 1.0, 8.8, 3.3, 1.15, "积压 $B_{st}$、截止期 $D$\n档位 $k∈\\{1,…,5\\}$", "算", "可延期任务")
    box(ax, 6.0, 8.85, 5.0, 2.45, "$N$ 卡，档位分配 $m_{rkjst}$\n$P^{\\mathrm{IT}}=Np^{\\mathrm{base}}+Σ(p^{\\mathrm{dyn}}_{rk}+p^{\\mathrm{lk}}_{j})m$\n多档 DVFS + 温区漏电", "算", "GPU 计算集群")
    box(ax, 12.6, 8.85, 3.9, 2.45, "$J^{\\mathrm{safe}}_{rk}$：$\\overline{T}_j+R^{\\mathrm{th}}h_{rk}$\n$≤T^{\\mathrm{chip,max}}$\n液冷温度 → 可用档位", "信", "热安全温区门控")
    arrow(ax, [(4.3, 10.7), (6.0, 10.7)], "算", "处理 $U^{\\mathrm{R}}$")
    arrow(ax, [(4.3, 9.4), (6.0, 9.4)], "算", "处理 $U^{\\mathrm{B}}$")
    arrow(ax, [(12.6, 10.1), (11.0, 10.1)], "信", "档位门控", loff=(0, 0.08))
    # 热管理层
    box(ax, 6.0, 5.65, 5.0, 2.2, "等效温度 $T^{w}_{st}$，热容 $C_w$\n温区 $j$（$u_{jst}$），式(9)–(12)", "热", "冷板式液冷环路")
    box(ax, 1.0, 6.55, 3.3, 1.2, "$q^{\\mathrm{FC}}≤\\phi_{jst}K^{\\mathrm{FC}}$", "冷", "干冷器自然冷却")
    box(ax, 1.0, 4.85, 3.3, 1.2, "$\\mathrm{COP}^{\\mathrm{MC}}_{jst}$，$K^{\\mathrm{MC}}$", "冷", "电制冷机组")
    box(ax, 12.6, 6.55, 3.9, 1.2, "$S^{\\mathrm{TES}}$，$E^{\\mathrm{TES}}$，$\\tau_{\\mathrm{TES}}$", "冷", "蓄冷水罐")
    box(ax, 12.6, 4.85, 3.9, 1.2, "$\\mathrm{COP}^{\\mathrm{HP}}_{j}$，$K^{\\mathrm{HP}}$ → 热用户", "热", "余热回收热泵")
    arrow(ax, [(8.5, 8.85), (8.5, 7.85)], "热", "散热 $\\beta P^{\\mathrm{IT}}$", loff=(0.08, -0.05), ha="left", va="top")
    arrow(ax, [(6.0, 7.15), (4.3, 7.15)], "冷", "$q^{\\mathrm{FC}}_{j}$", lpos=0.72)
    arrow(ax, [(6.0, 6.1), (4.3, 5.45)], "冷", "$q^{\\mathrm{MC}}_{j}$", lpos=0.75)
    arrow(ax, [(11.0, 7.15), (12.6, 7.15)], "冷", "$q^{\\mathrm{ch}}/q^{\\mathrm{dis}}$", both=True, loff=(0, -0.12), va="top")
    arrow(ax, [(11.0, 6.1), (12.6, 5.45)], "热", "$q^{\\mathrm{HP}}_{j}$")
    arrow(ax, [(10.8, 7.85), (10.8, 8.45), (12.2, 8.45), (12.2, 9.4), (12.6, 9.4)], "信", "$T^{w}$", lpos=0.35, loff=(0, 0.06))
    # 能源层
    box(ax, 6.0, 1.4, 5.0, 1.6, "式(24)：购售电 + 风光 + 储能\n= IT + 冷源 + 热泵 + 辅助", "电", "园区电母线")
    box(ax, 1.0, 2.55, 3.3, 1.15, "$K^{\\mathrm{PV}}$、$K^{\\mathrm{WT}}$，弃电 $P^{\\mathrm{cur}}$", "电", "光伏 / 风电")
    box(ax, 1.0, 0.85, 3.3, 1.15, "$P^{\\mathrm{B}}$、$E^{\\mathrm{B}}$", "电", "电池储能")
    box(ax, 12.6, 2.55, 3.9, 1.15, "$P^{\\mathrm{buy}}$ / $P^{\\mathrm{sell}}$，分时电价", "电", "上级电网")
    box(ax, 12.6, 0.85, 3.9, 1.15, "阶梯碳价，售热收益", "信", "碳交易 / 热市场")
    arrow(ax, [(4.3, 3.1), (6.0, 2.6)], "电"); arrow(ax, [(4.3, 1.4), (6.0, 1.9)], "电", both=True)
    arrow(ax, [(12.6, 3.1), (11.0, 2.6)], "电", both=True)
    arrow(ax, [(8.5, 3.0), (8.5, 5.65)], "电", "冷源/泵/热泵用电", lpos=0.62, loff=(0.1, 0), ha="left", va="center")
    arrow(ax, [(6.3, 3.0), (6.3, 3.55), (5.7, 3.55), (5.7, 8.55), (6.6, 8.55), (6.6, 8.85)], "电")
    ax.text(5.62, 4.3, "IT 供电 $P^{\\mathrm{IT}}$", rotation=90, ha="right", va="bottom", fontsize=6.8, color=EC["电"])
    hs = [plt.Line2D([], [], color=EC[k], ls=LS[k], lw=1.0) for k in ("电", "冷", "热", "算", "信")]
    ax.legend(hs, ["电功率流", "冷量流", "热量流", "算力任务流", "信息/约束"], loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=5, frameon=False, fontsize=7.2)
    save(fig, "图1_系统结构")


def fig2():
    W, H = 15.5, 10.0; fig, ax = canvas(W, H)
    nodes = {"f": (7.75, 8.6, "频率档位 $k$\n（$m_{rkjst}$）", "算", 3.6, 1.2), "p": (12.9, 6.0, "GPU 功率 $P^{\\mathrm{IT}}$\n动态 $p^{\\mathrm{dyn}}_{rk}$ + 漏电 $p^{\\mathrm{lk}}_{j}$", "电", 4.2, 1.3),
             "h": (11.0, 2.2, "发热 $\\beta P^{\\mathrm{IT}}$", "热", 3.0, 1.0), "T": (4.5, 2.2, "液冷温度 $T^{w}$\n热惯性 $C_w$", "热", 3.4, 1.2),
             "c": (2.6, 6.0, "冷却与余热\n$\\phi_{jst}$、$\\mathrm{COP}^{\\mathrm{MC}}_{jst}$、$\\mathrm{COP}^{\\mathrm{HP}}_{j}$", "冷", 4.4, 1.3)}
    for k, (x, y, t, kd, w, h) in nodes.items(): box(ax, x - w / 2, y - h / 2, w, h, t, kd, fs=7.6)
    arrow(ax, [(9.55, 8.45), (12.0, 6.65)], "电", "式(3)", loff=(0.15, 0.05), ha="left")
    arrow(ax, [(12.6, 5.35), (11.4, 2.7)], "热", "式(9)", loff=(0.15, 0), ha="left", va="center")
    arrow(ax, [(9.5, 2.2), (6.2, 2.2)], "热", "式(9)", loff=(0, 0.1))
    arrow(ax, [(3.6, 2.8), (2.1, 5.35)], "信", "依赖 $T^{w}$：式(13)(16)(18)", loff=(-0.15, 0), ha="right", va="center")
    arrow(ax, [(3.3, 5.35), (4.8, 2.8)], "冷", "移热 $q^{\\mathrm{FC}},q^{\\mathrm{MC}},q^{\\mathrm{HP}}$", loff=(0.15, 0), ha="left", va="center")
    arrow(ax, [(5.6, 2.8), (7.3, 8.0)], "信", "门控 式(11)(12)：$T^{w}$↑，高频档禁用", lpos=0.9, loff=(-0.2, 0), ha="right", va="center")
    arrow(ax, [(6.2, 2.6), (11.6, 5.35)], "信", "漏电 式(4)：$T^{w}$↑，$p^{\\mathrm{lk}}$↑", lpos=0.42, loff=(0.1, -0.12), ha="left", va="top")
    ax.text(0.4, 0.75, "升温收益：$\\phi$↑，$\\mathrm{COP}^{\\mathrm{MC}}$↑，$\\mathrm{COP}^{\\mathrm{HP}}$↑", fontsize=7.6, color=EC["冷"], ha="left")
    ax.text(0.4, 0.3, "升温代价：漏电功率↑，高频档位被门控禁用", fontsize=7.6, color=EC["热"], ha="left")
    save(fig, "图2_闭环耦合")


def fig3():
    W, H = 12.5, 15.0; fig, ax = canvas(W, H); cx, bw = 5.4, 6.4
    def b(y, t, kd="信", h=1.05): box(ax, cx - bw / 2, y, bw, h, t, kd, fs=7.4)
    b(13.7, "输入：典型日场景 $\\xi_s$、经验概率 $p^0$、半径 $\\theta_1,\\theta_∞$\n初始化 $LB=-∞$，$UB=+∞$，$k=1$，$p^{(1)}=p^0$", "无")
    b(12.1, "求解主问题(35)：保留全部典型日运行变量与概率割\n得到 $x^{(k)}$、$\\eta^{(k)}$，按式(38)更新 $LB$", "算")
    b(10.5, "子问题第一步(36)：固定 $x^{(k)}$\n$K$ 个典型日运行 MILP 并行求解，得到 $Q_s(x^{(k)})$", "电")
    b(8.9, "子问题第二步(37)：在模糊集 $P$ 内求最恶劣分布 $p^{*}$\n（线性规划），按式(38)更新 $UB$", "冷")
    ax.add_patch(plt.Polygon([(cx, 8.2), (cx + 2.6, 7.35), (cx, 6.5), (cx - 2.6, 7.35)], closed=True, fc="white", ec="#555", lw=0.8, zorder=2))
    ax.text(cx, 7.35, "$(UB-LB)/UB≤ε$？", ha="center", va="center", fontsize=7.6, zorder=3)
    b(4.9, "输出最优容量 $x^{*}$ 与各典型日运行方案", "无")
    box(ax, 9.3, 9.95, 2.9, 1.25, "加入概率割\n$\\eta≥Σ_s p^{*}_s d^{T}y_s$\n$k← k+1$", "信", fs=7.2)
    for y0, y1 in ((13.7, 13.15), (12.1, 11.55), (10.5, 9.95), (8.9, 8.2), (6.5, 5.95)): arrow(ax, [(cx, y0), (cx, y1)], "信")
    ax.text(cx + 0.1, 6.15, "是", fontsize=7.4, ha="left")
    arrow(ax, [(cx + 2.6, 7.35), (10.75, 7.35), (10.75, 9.95)], "信", "否", lpos=0.2, loff=(0, 0.1))
    arrow(ax, [(10.75, 11.2), (10.75, 12.62), (cx + bw / 2, 12.62)], "信")
    ax.text(0.4, 3.9, "SP：$\\theta_1=\\theta_∞=0$（模糊集退化为 $p^0$）\nRO：$\\theta_1=2$，$\\theta_∞=1$（模糊集为概率单纯形）\nDRO：$\\theta_1,\\theta_∞$ 由式(31)按置信水平确定",
            fontsize=7.4, ha="left", va="top", linespacing=1.5, bbox=dict(fc="#F7F7F7", ec="#BBBBBB", lw=0.6, pad=4))
    ax.set_ylim(2.2, H); fig.set_size_inches(W / 2.54, (H - 2.2) / 2.54)
    save(fig, "图3_CCG流程")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
