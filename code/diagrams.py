"""
论文示意图 (矢量 PDF/SVG + 600 dpi PNG):
  F1 系统结构图 —— 电能流、热量流、算力任务流、信息/约束耦合, 模块与变量与式(1)–(18)一致
  F2 研究框架图 —— 研究问题 → 模型 → 方法 → 实验 → 输出
  F3 求解流程图 —— 数据输入 → 情景与模糊集 → 模型构建 → MILP 分支定界迭代与收敛判断 → 重调度/校验 → 输出
坐标单位为 cm; 字体: 西文/公式 Times New Roman, 中文宋体 (plot_style 自动合并)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import plot_style as PS
import config as C

OUT = C.FIG / "diagrams"; OUT.mkdir(parents=True, exist_ok=True)
PS._fonts(8)
plt.rcParams.update({"font.size": 8, "lines.linewidth": 1.0})
LW_BOX, LW_ARR = 0.8, 1.0
EC = {"电": "#3C5488", "热": "#C0392B", "算": "#00876C", "信": "#6E6E6E"}       # 线色: 电能/热量/算力任务/信息
FC_ = {"电": "#EDF1F9", "热": "#FBECE9", "算": "#E6F3EF", "信": "#F4F4F4", "无": "white"}


def canvas(w, h):
    fig = plt.figure(figsize=(w / 2.54, h / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", kind="无", title=None, fs=7.5, ec=None, ls="-", lw=LW_BOX, r=0.12, tfs=8, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ls=ls,
                                ec=ec or EC.get(kind, "#555555"), fc=FC_[kind], zorder=2))
    if title:
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=tfs, fontweight="bold", zorder=3)
        ty = y + (h - 0.45) / 2
    else:
        ty = y + h / 2
    if text:
        xx = x + w / 2 if align == "center" else x + 0.15
        ax.text(xx, ty, text, ha=align, va="center", fontsize=fs, linespacing=1.35, zorder=3)


def arrow(ax, pts, kind, label=None, lpos=0.5, loff=(0, 0.12), ha="center", va="bottom", fs=7, both=False, ls=None,
          head=True):
    pts = np.asarray(pts, float); col = EC[kind]; ls = ls or ("--" if kind == "信" else "-")
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=col, lw=LW_ARR, ls=ls, zorder=1, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=col, lw=LW_ARR, ls=ls,
                                    shrinkA=0, shrinkB=0))
    if both:
        ax.annotate("", xy=pts[0], xytext=pts[1], zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=col, lw=LW_ARR,
                                    shrinkA=0, shrinkB=0))
    if label:
        seg = np.diff(pts, axis=0); L = np.hypot(*seg.T); s = lpos * L.sum(); i = int(np.searchsorted(np.cumsum(L), s))
        i = min(i, len(seg) - 1); f = (s - np.r_[0, np.cumsum(L)][i]) / max(L[i], 1e-9)
        p = pts[i] + f * seg[i]
        ax.text(p[0] + loff[0], p[1] + loff[1], label, ha=ha, va=va, fontsize=fs, color=col, zorder=3)


def save(fig, name):
    for ext in ("pdf", "svg"): fig.savefig(OUT / f"{name}.{ext}")
    fig.savefig(OUT / f"{name}.png", dpi=600); plt.close(fig)


# ============================== F1 系统结构图 ==============================
def fig_system():
    W, H = 17.6, 12.6
    fig, ax = canvas(W, H)
    # ---- 顶部: 外生情景与第一阶段容量 ----
    box(ax, 0.3, 11.15, 8.6, 1.2, "负荷 $A^{\\mathrm{R}}_{dst},A^{\\mathrm{B}}_{dst}$；光伏出力系数 $g^{\\mathrm{PV}}_{dst}$；"
        "环境温度 $T_{\\mathrm{amb},dst}$\n有限支持情景 $s\\in𝒮_d$，概率 $\\boldsymbol{p}_d\\in𝒫_d$（式17）",
        "信", title="外生不确定情景 $\\xi_{ds}$", fs=7)
    box(ax, 9.3, 11.15, 8.0, 1.2, "$\\boldsymbol{x}=(N,K^{\\mathrm{PV}},P^{\\mathrm{B}},E^{\\mathrm{B}},K^{\\mathrm{FC}},"
        "K^{\\mathrm{MC}})$（式14–15）\n各设备框中对应容量变量；第二阶段运行量 $\\boldsymbol{y}_{ds}$（式1–13）",
        "信", title="第一阶段容量决策", fs=7)
    # ---- 电源侧与母线 ----
    box(ax, 0.3, 8.2, 3.0, 1.75, "$\\mathrm{0}\\leq P^{\\mathrm{grid}}_{dst}\\leq P^{\\mathrm{grid,max}}$\n分时电价 $c^{e}_{dt}$（式16）",
        "电", title="外部电网", fs=7)
    box(ax, 0.3, 5.55, 3.0, 1.75, "容量 $K^{\\mathrm{PV}}$\n$P^{\\mathrm{PV}}_{dst}\\leq K^{\\mathrm{PV}}g^{\\mathrm{PV}}_{dst}$（式10）",
        "电", title="光伏", fs=7)
    box(ax, 0.3, 2.65, 3.0, 2.0, "$P^{\\mathrm{B}},E^{\\mathrm{B}}$；电量 $S_{dst}$\n$P^{\\mathrm{ch}}_{dst},P^{\\mathrm{dis}}_{dst}$，"
        "互斥 $b_{dst}$\n（式11–12）", "电", title="电池储能", fs=7)
    ax.add_patch(Rectangle((4.0, 0.55), 0.2, 9.35, fc=EC["电"], ec="none", zorder=2))
    ax.text(4.1, 10.0, "交流母线\n（式13）", ha="center", va="bottom", fontsize=7, color=EC["电"], linespacing=1.15)
    arrow(ax, [(3.3, 9.1), (4.0, 9.1)], "电", "$P^{\\mathrm{grid}}$", loff=(0, 0.08))
    arrow(ax, [(3.3, 6.4), (4.0, 6.4)], "电", "$P^{\\mathrm{PV}}$", loff=(0, 0.08))
    arrow(ax, [(3.3, 3.65), (4.0, 3.65)], "电", "$P^{\\mathrm{ch}}$/$P^{\\mathrm{dis}}$", loff=(0, -0.1), va="top", both=True, fs=6.2)
    # ---- 计算任务 → GPU 集群 ----
    box(ax, 4.9, 9.75, 4.1, 1.25, "实时：$U^{\\mathrm{R}}_{dst}=A^{\\mathrm{R}}_{dst}$（式2）\n延期：$A^{\\mathrm{B}}_{dst}\\!\\rightarrow\\!B_{dst}"
        "\\!\\rightarrow\\!U^{\\mathrm{B}}_{dst}$，期限 $D$（式2–3）", "算", title="计算任务", fs=6.6, tfs=7.5)
    arrow(ax, [(6.95, 9.75), (6.95, 9.4)], "算")
    box(ax, 4.9, 5.55, 4.1, 3.85, "", "算", title="GPU 集群（式1）", )
    for i, (nm, v) in enumerate([("$k_1$ 标称", "$v_{rk},p_{rk}$"), ("$k_2$ 浅锁频", ""), ("$k_3$ 深锁频（仅B）", "")]):
        box(ax, 5.15, 8.25 - i * 0.55, 3.6, 0.45, nm, "无", ec=EC["算"], fs=6.8, r=0.06, lw=0.6)
    ax.text(6.95, 6.35, "配置 $N$；模式分配 $m_{rkdst}$（$v_{rk},p_{rk}$）\n休眠功率 $p^{\\mathrm{sl}}$；IT功率 $P^{\\mathrm{IT}}_{dst}$",
            ha="center", va="center", fontsize=7, linespacing=1.35)
    arrow(ax, [(4.2, 7.55), (4.9, 7.55)], "电", "$\\kappa_{\\mathrm{aux}}P^{\\mathrm{IT}}$", loff=(0.02, 0.08), fs=6.6)
    # ---- 空气侧与其他辅助负荷 ----
    box(ax, 4.9, 2.95, 4.1, 1.75, "$(\\mathrm{1}-\\beta_{\\mathrm{liq}})P^{\\mathrm{IT}}$，电耗系数 $a^{\\mathrm{air}}$\n"
        "（电耗计入 $\\kappa_{\\mathrm{aux}}$，式13）", "热", title="空气侧散热", fs=7)
    arrow(ax, [(6.95, 5.55), (6.95, 4.7)], "热", "$(\\mathrm{1}-\\beta_{\\mathrm{liq}})P^{\\mathrm{IT}}$", loff=(0.1, -0.05),
          ha="left", va="center", fs=6.6)
    arrow(ax, [(9.0, 3.4), (9.75, 3.4)], "热")
    ax.text(9.38, 3.5, "排至\n环境", ha="center", va="bottom", fontsize=6.5, color=EC["热"], linespacing=1.1)
    box(ax, 4.9, 0.55, 4.1, 1.0, "其他辅助负荷 $P^{\\mathrm{0}}_{dst}$", "电", fs=7)
    arrow(ax, [(4.2, 1.05), (4.9, 1.05)], "电")
    # ---- 液冷环路 ----
    box(ax, 10.3, 6.35, 7.0, 3.6, "", "热", title="液冷环路（式4–5）")
    ax.text(13.8, 9.05, "等效热容 $C_w$；等效温度 $T^w_{dst}$；温区选择 $u_{jdst}\\in\\{\\mathrm{0,1}\\}$", ha="center", va="center",
            fontsize=7)
    zones = C.ZONES; x0, wz = 10.75, 6.1 / len(zones)
    zc = ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#FDD0A2", "#FCBBA1"]
    for i, (lo, hi) in enumerate(zones):
        ax.add_patch(Rectangle((x0 + i * wz, 7.65), wz, 0.6, fc=zc[i], ec="#888888", lw=0.5, zorder=3))
        ax.text(x0 + (i + 0.5) * wz, 7.95, f"{lo:g}–{hi:g}", ha="center", va="center", fontsize=6.5, zorder=4)
    ax.text(x0 - 0.05, 7.95, "$j$", ha="right", va="center", fontsize=7)
    ax.text(13.8, 7.45, "温区 $[\\underline{T}_j,\\overline{T}_j]$ / ℃", ha="center", va="top", fontsize=6.5)
    ax.text(13.8, 6.75, "热点约束：$\\overline{T}_j+R^{\\mathrm{th}}_{rk}h_{rk}\\leq T^{\\mathrm{chip,max}}$（式6）",
            ha="center", va="center", fontsize=6.8)
    arrow(ax, [(9.0, 9.0), (10.3, 9.0)], "热", "$\\beta_{\\mathrm{liq}}P^{\\mathrm{IT}}$", loff=(0, 0.08), fs=6.6)
    arrow(ax, [(10.3, 6.9), (9.0, 6.9)], "信", "$u_{jdst}\\!\\rightarrow\\!𝒥^{\\mathrm{safe}}_{rk}$",
          loff=(0, 0.1), fs=6.2)
    ax.text(9.65, 6.78, "模式门控\n（式7）", ha="center", va="top", fontsize=6.2, color=EC["信"], linespacing=1.1)
    # ---- 冷源与环境 ----
    box(ax, 10.3, 2.95, 3.1, 2.3, "容量 $K^{\\mathrm{FC}}$\n$q^{\\mathrm{FC}}_{jdst}\\leq\\underline{\\phi}\\,_{jdst}K^{\\mathrm{FC}}$\n"
        "电耗 $a^{\\mathrm{FC}}_{jdst}$（式8）", "热", title="自然冷却", fs=6.8)
    box(ax, 14.2, 2.95, 3.1, 2.3, "容量 $K^{\\mathrm{MC}}$\n$q^{\\mathrm{MC}}_{jdst}\\leq K^{\\mathrm{MC}}$\n"
        "电耗 $a^{\\mathrm{MC}}_{jdst}$（式9）", "热", title="机械制冷", fs=6.8)
    arrow(ax, [(11.85, 6.35), (11.85, 5.25)], "热", "$q^{\\mathrm{FC}}_{jdst}$", loff=(0.08, 0), ha="left", va="center", fs=6.6)
    arrow(ax, [(15.75, 6.35), (15.75, 5.25)], "热", "$q^{\\mathrm{MC}}_{jdst}$", loff=(0.08, 0), ha="left", va="center",
          fs=6.6)
    box(ax, 13.55, 2.95, 0.5, 2.3, "", "无", ec="#888888", r=0.08)
    ax.text(13.8, 4.1, "室外环境", rotation=90, ha="center", va="center", fontsize=6.8)
    arrow(ax, [(13.4, 3.55), (13.55, 3.55)], "热"); arrow(ax, [(14.2, 3.55), (14.05, 3.55)], "热")
    ax.text(13.8, 2.8, "$T_{\\mathrm{amb},dst}\\!\\rightarrow\\!\\underline{\\phi}\\,_{jdst},a^{\\mathrm{MC}}_{jdst}$",
            ha="center", va="top", fontsize=6.4, color=EC["信"])
    # 母线 → 冷源电耗 (走线位于空气侧框与辅助负荷框之间)
    arrow(ax, [(4.2, 2.25), (11.0, 2.25), (11.0, 2.95)], "电")
    arrow(ax, [(11.0, 2.25), (16.6, 2.25), (16.6, 2.95)], "电")
    ax.text(7.3, 2.33, "$\\sum_j a^{\\mathrm{FC}}_{jdst}q^{\\mathrm{FC}}_{jdst}$、$\\sum_j a^{\\mathrm{MC}}_{jdst}q^{\\mathrm{MC}}_{jdst}$",
            ha="center", va="bottom", fontsize=6.4, color=EC["电"])
    ax.plot([11.0], [2.25], "o", ms=2.2, color=EC["电"], zorder=3)
    # ---- 图例 ----
    lx, ly = 10.3, 1.25
    for i, (k, t) in enumerate([("电", "电能流"), ("热", "热量流"), ("算", "算力任务流"), ("信", "信息/约束耦合")]):
        xx = lx + [0, 1.6, 3.2, 5.15][i]
        arrow(ax, [(xx, ly), (xx + 0.5, ly)], k); ax.text(xx + 0.58, ly, t, va="center", fontsize=6.8)
    ax.text(lx, 0.62, "注：$\\kappa_{\\mathrm{aux}}=\\mathrm{1}+\\delta_e+a^{\\mathrm{circ}}\\beta_{\\mathrm{liq}}+a^{\\mathrm{air}}"
            "(\\mathrm{1}-\\beta_{\\mathrm{liq}})$，\n含配电损耗、主循环泵与空气侧电耗（式13）。", fontsize=6.4, va="center", linespacing=1.3)
    save(fig, "F1_系统结构图")


# ============================== F2 研究框架图 ==============================
def fig_framework():
    W, H = 17.6, 11.0
    fig, ax = canvas(W, H)
    cols = [(0.3, "研究问题"), (4.75, "数学模型"), (9.2, "求解方法"), (13.65, "实验验证")]
    cw = 3.65
    for x, t in cols:
        ax.add_patch(FancyBboxPatch((x, 9.95), cw, 0.65, boxstyle="round,pad=0,rounding_size=0.1", fc="#3C5488",
                                    ec="none", zorder=2))
        ax.text(x + cw / 2, 10.27, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=3)
    for i in range(3):
        x = cols[i][0] + cw
        arrow(ax, [(x + 0.08, 10.27), (cols[i + 1][0] - 0.08, 10.27)], "电")
    Q = [("Q1 算力调节", "GPU 运行模式改变处理速率与\n功耗，如何进入容量规划？"),
         ("Q2 热—算耦合", "液冷温度如何反向限制 GPU\n模式，并决定冷源配置？"),
         ("Q3 不确定性", "负荷—气象—光伏联合分布\n难以准确估计时如何配置？")]
    Mm = [("GPU 模式与任务模型", "式(1)–(3)：$m_{rkdst}$，$U^r_{dst}$，$B_{dst}$"),
          ("液冷热动态与热安全", "式(4)–(7)：$T^w_{dst}$，$u_{jdst}$，$𝒥^{\\mathrm{safe}}_{rk}$"),
          ("冷源·光伏·储能·功率平衡", "式(8)–(13)：$q^{\\mathrm{FC}},q^{\\mathrm{MC}},P^{\\mathrm{grid}},S_{dst}$"),
          ("两阶段有限支持矩 DRO", "式(14)–(18)：$\\boldsymbol{x}$，$𝒫_d$，$Q_{ds}(\\boldsymbol{x})$")]
    Me = [("数据驱动情景", "BurstGPT 轨迹、典型年气象、\n分时电价 → 联合日\nk-medoids → $𝒮_d,\\hat{p}_d$"),
          ("模糊集估计", "bootstrap 估计 $\\delta_d,v^{\\pm}_{d\\ell}$\n验证集校准\n非空性 LP 检验"),
          ("精确重构与求解", "内层最大化 LP 强对偶\n→ 单一 MILP；有效不等式\nHiGHS 分支定界"),
          ("交叉验证与后处理", "割平面迭代（第5节）\n验证等价性；固定 $\\boldsymbol{x}$\n重调度 $Q_{ds}(\\boldsymbol{x})$")]
    Ex = [("基准与消融", "Baseline、M1 调频\nM2 热耦合、M3"),
          ("概率模型对比", "M3-SP、M4-DRO\nM5-RO、M6 极端支持"),
          ("物理一致性校验", "功率/热平衡残差、热点\n温度、SOC、任务完成"),
          ("样本外与分布偏移", "123 个测试日\n指数倾斜 $\\theta$"),
          ("敏感性分析", "$\\kappa$、峰谷浮动\n$R^{\\mathrm{th}}$、$\\beta_{\\mathrm{liq}}$、$C_w$")]
    def stack(x, items, kind, top=9.65, bottom=1.95, fs=6.8):
        n = len(items); g = 0.18; h = (top - bottom - g * (n - 1)) / n; ys = []
        for i, (t, s) in enumerate(items):
            y = top - (i + 1) * h - i * g
            box(ax, x, y, cw, h, s, kind, title=t, fs=fs, tfs=7.6); ys.append(y + h / 2)
        return ys, h
    yq, _ = stack(cols[0][0], Q, "信", fs=6.8); ym, _ = stack(cols[1][0], Mm, "电", fs=6.6)
    ye, _ = stack(cols[2][0], Me, "热", fs=6.6); yx, _ = stack(cols[3][0], Ex, "算", fs=6.6)
    # 模型 → 方法 → 实验: 列之间整体衔接 (逐项之间不存在一一对应, 故不画逐项连线)
    for i in (1, 2):
        xa = cols[i][0] + cw; xb = cols[i + 1][0]
        ax.add_patch(FancyBboxPatch((xa + 0.12, 2.0), xb - xa - 0.24, 7.6, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#F4F4F4", ec="none", zorder=0))
        arrow(ax, [(xa + 0.18, 5.8), (xb - 0.18, 5.8)], "信", ls="-")
    # 问题 → 模型 的对应关系
    for a, bs in [(0, [0]), (1, [1, 2]), (2, [3])]:
        for b in bs:
            arrow(ax, [(cols[0][0] + cw, yq[a]), (cols[1][0], ym[b])], "信", ls="-")
    # 底部输出
    box(ax, 0.3, 0.3, 17.0, 1.35, "第一阶段容量 $\\boldsymbol{x}^*=(N,K^{\\mathrm{PV}},P^{\\mathrm{B}},E^{\\mathrm{B}},K^{\\mathrm{FC}},"
        "K^{\\mathrm{MC}})$；情景运行策略 $\\boldsymbol{y}^*_{ds}$（GPU 模式、液冷温区、冷源出力、购电与储能）；最坏分布 $\\boldsymbol{p}^*_d$；"
        "可靠性与经济性评价", "信", title="输出", fs=7)
    for x, _ in cols:
        arrow(ax, [(x + cw / 2, 1.95), (x + cw / 2, 1.65)], "信", ls="-")
    save(fig, "F2_研究框架图")


# ============================== F3 求解流程图 ==============================
def fig_flow():
    W, H = 17.6, 15.6
    fig, ax = canvas(W, H)
    cx, bw = 6.15, 8.4; x0 = cx - bw / 2
    steps = [
        ("数据输入", "BurstGPT 请求轨迹（$A^{\\mathrm{R}},A^{\\mathrm{B}}$）、EPW 典型年气象、分时电价 $c^{e}_{dt}$、设备参数表", "信"),
        ("预处理", "调度日窗口（6:00 起）与规模化；PVWatts 计算 $g^{\\mathrm{PV}}_{dst}$；\n温区系数 $\\underline{\\phi}\\,_{jdst}$、"
                   "制冷电耗 $a^{\\mathrm{MC}}_{jdst}$；安全模式集 $𝒥^{\\mathrm{safe}}_{rk}$（式6）", "信"),
        ("情景构建", "季节类别 $d$，联合日（气象×负荷）；训练/测试划分；\nk-medoids 得有限支持 $𝒮_d$ 与经验概率 $\\hat{p}_{ds}$", "电"),
        ("模糊集估计（式17）", "$\\hat{\\mu}_d$、$\\rho_{\\ell,ds}$；bootstrap 得 $\\delta^{\\mathrm{0}}_d,v^{\\mathrm{0}\\pm}_{d\\ell}$；"
                             "验证集校准 $\\kappa_\\delta,\\kappa_v$", "电"),
        None,                                                     # 判断 1
        ("模型构建", "第一阶段 $\\boldsymbol{x}$（式14–15）+ 情景运行约束（式1–13,16）\n内层最大化（$\\boldsymbol{p}_d\\in𝒫_d$）"
                    "以 LP 对偶替换 → 单一 MILP（式18）", "热"),
        ("MILP 求解（分支定界）", "LP 松弛 → 分支 → 更新可行上界与对偶下界", "热"),
        None,                                                     # 判断 2
        ("重调度与校验", "固定 $\\boldsymbol{x}^*$ 逐情景求 $Q_{ds}(\\boldsymbol{x}^*)$；校验式(4)(13)残差、"
                         "热点温度、SOC、任务完成", "算"),
        ("评估与输出", "样本外/分布偏移评估、敏感性分析；输出 $\\boldsymbol{x}^*,\\boldsymbol{y}^*_{ds},\\boldsymbol{p}^*_d$", "算"),
    ]
    hs = [0.95, 1.2, 1.2, 1.0, 1.0, 1.2, 0.9, 1.0, 1.2, 1.0]
    y = 15.0; centers = []
    for st, h in zip(steps, hs):
        y -= h
        if st is None:
            centers.append((y + h / 2, h)); y -= 0.42; continue
        box(ax, x0, y, bw, h, st[1], st[2], title=st[0], fs=6.9, tfs=7.8)
        centers.append((y + h / 2, h)); y -= 0.42
    def diamond(yc, text, w=5.2, h=1.0):
        pts = np.array([(cx, yc + h / 2), (cx + w / 2, yc), (cx, yc - h / 2), (cx - w / 2, yc), (cx, yc + h / 2)])
        ax.fill(pts[:, 0], pts[:, 1], fc="#FFF8E5", ec="#B7950B", lw=LW_BOX, zorder=2)
        ax.text(cx, yc, text, ha="center", va="center", fontsize=7, zorder=3)
    diamond(centers[4][0], "$𝒫_d\\neq\\varnothing$？（非空性 LP）")
    diamond(centers[7][0], "绝对间隙 $\\leq\\varepsilon$ 或达到时限？")
    # 主干箭头
    for i in range(len(steps) - 1):
        yc, h = centers[i]; yn, hn = centers[i + 1]
        top_next = yn + (0.5 if steps[i + 1] is None else hn / 2)
        bot = yc - (0.5 if steps[i] is None else h / 2)
        arrow(ax, [(cx, bot), (cx, top_next)], "电")
    ax.text(cx + 0.12, centers[4][0] - 0.62, "是", fontsize=7, va="top")
    ax.text(cx + 0.12, centers[7][0] - 0.62, "是", fontsize=7, va="top")
    # 判断1 否: 最小放大 τ 后回到模糊集
    yd = centers[4][0]
    box(ax, 0.15, yd - 0.45, 1.3, 0.9, "最小统一\n放大 $\\tau$", "电", fs=6.8)
    arrow(ax, [(cx - 2.6, yd), (1.45, yd)], "电"); ax.text(1.65, yd + 0.08, "否", fontsize=7, va="bottom")
    arrow(ax, [(0.8, yd + 0.45), (0.8, centers[3][0]), (x0, centers[3][0])], "电")
    # 判断2 否: 回到分支定界 (迭代)
    yd2 = centers[7][0]; ys = centers[6][0]
    arrow(ax, [(cx - 2.6, yd2), (0.8, yd2), (0.8, ys), (x0, ys)], "热")
    ax.text(0.95, yd2 + 0.08, "否：继续分支", fontsize=7, va="bottom")
    # ---- 右侧: 割平面交叉验证 (论文第5节迭代算法) ----
    rx, rw = 11.15, 5.6
    ytop = centers[5][0] + 0.6
    ax.add_patch(FancyBboxPatch((rx - 0.2, centers[8][0] - 0.75), rw + 0.4, ytop - centers[8][0] + 1.25,
                                boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec="#888888", lw=0.7, ls="--"))
    ax.text(rx + rw / 2, ytop + 0.35, "交叉验证：逐次加入最坏分布约束（论文第5节）", ha="center", va="center", fontsize=7.5,
            fontweight="bold")
    yb = [ytop - 0.15 - 1.3, ytop - 0.15 - 2.95, ytop - 0.15 - 4.5]
    box(ax, rx, yb[0], rw, 1.3, "$\\min C^{\\mathrm{plan}}+\\sum_d n_d\\eta_d$，$\\eta_d\\geq\\sum_s p^{(i)}_{ds}C^{\\mathrm{op}}_{ds}$"
        "，$i=\\mathrm{1},\\dots,k$", "热", title="主问题（MILP）", fs=6.6, tfs=7.5)
    box(ax, rx, yb[1], rw, 1.3, "$\\boldsymbol{p}^{(k+\\mathrm{1})}_d$ 使 $\\sum_s p_{ds}C^{\\mathrm{op}}_{ds}(\\boldsymbol{x}^{(k)})$ "
        "在 $𝒫_d$ 上最大", "电", title="子问题（LP）", fs=6.6, tfs=7.5)
    arrow(ax, [(rx + rw / 2, yb[0]), (rx + rw / 2, yb[1] + 1.3)], "电", "$\\boldsymbol{x}^{(k)},C^{\\mathrm{op}}$",
          loff=(0.1, 0), ha="left", va="center", fs=6.6)
    yc3 = yb[2] + 0.5
    pts = np.array([(rx + rw / 2, yc3 + 0.5), (rx + rw / 2 + 2.6, yc3), (rx + rw / 2, yc3 - 0.5),
                    (rx + rw / 2 - 2.6, yc3), (rx + rw / 2, yc3 + 0.5)])
    ax.fill(pts[:, 0], pts[:, 1], fc="#FFF8E5", ec="#B7950B", lw=LW_BOX, zorder=2)
    ax.text(rx + rw / 2, yc3, "$\\sum_d n_d(\\boldsymbol{p}^{(k+\\mathrm{1})\\top}_d C^{\\mathrm{op}}_d-\\eta_d)\\leq\\varepsilon$？",
            ha="center", va="center", fontsize=6.8, zorder=3)
    arrow(ax, [(rx + rw / 2, yb[1]), (rx + rw / 2, yc3 + 0.5)], "电")
    arrow(ax, [(rx + rw / 2 + 2.6, yc3), (rx + rw + 0.15, yc3), (rx + rw + 0.15, yb[0] + 0.65), (rx + rw, yb[0] + 0.65)], "电")
    ax.text(rx + rw + 0.22, (yc3 + yb[0]) / 2, "否：加入割平面", rotation=90, fontsize=6.6, ha="left", va="center")
    arrow(ax, [(rx + rw / 2, yc3 - 0.5), (rx + rw / 2, centers[8][0] + 0.15)], "电")
    box(ax, rx, centers[8][0] - 0.55, rw, 0.7, "是：与对偶重构目标值、容量比较（verify.py）", "无", ec="#888888", fs=6.8)
    arrow(ax, [(x0 + bw, centers[5][0]), (rx - 0.2, centers[5][0])], "信")
    ax.text((x0 + bw + rx - 0.2) / 2, centers[5][0] - 0.1, "同一\n模型", fontsize=6.4, ha="center", va="top", color=EC["信"],
            linespacing=1.1)
    save(fig, "F3_求解流程图")


if __name__ == "__main__":
    fig_system(); fig_framework(); fig_flow()
    print("written", sorted(p.name for p in OUT.glob("*.png")))
