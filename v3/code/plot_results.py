"""
图5–图16（执行方案 5.3–5.6 节）。数据缺失的图自动跳过，结果补齐后重新运行即可。
用法：python plot_results.py [site]          输出 v3/figures/图x_*.png / .pdf
"""
import os, sys, glob, pickle
import numpy as np, pandas as pd
import plot_style as PS
import matplotlib.pyplot as plt
import params as PR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"; FIG = f"{ROOT}/figures"; os.makedirs(FIG, exist_ok=True)
PS.style_b()
C = dict(S1="#7F7F7F", S2="#4DBBD5", S3="#00A087", S4="#3C5488", S5="#E64B35", S6="#8491B4", S7="#F39B7F")
NAME = dict(S1="S1", S2="S2", S3="S3", S4="S4", S5="S5", S6="S6(SP)", S7="S7(RO)")
TAGS = dict(S1="DRO_S1", S2="DRO_S2", S3="DRO_S3", S4="DRO_S4", S5="DRO_S5", S6="SP_S6", S7="RO_S7")


def load(path):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("图已保存", name)


def label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, va="top", ha="left", fontsize=9)


# ---------------------------------------------------------------- 图5 容量
def fig5(site):
    rs = {s: load(f"{RES}/{site}_{TAGS[s]}.pkl") for s in ("S1", "S2", "S3", "S4", "S5")}
    if any(v is None for v in rs.values()): return
    fig, axs = plt.subplots(2, 2, figsize=(17 * PS.CM, 10 * PS.CM))
    groups = [("(a) 冷源容量/MW", [("K_FC", "FC", "自然冷却"), ("K_MC", "MC", "机械制冷")]),
              ("(b) 储能/MW·MWh", [("P_B", "PB", "储能功率"), ("E_B", "EB", "储能容量")]),
              ("(c) 蓄冷/MWh", [("E_TES", "TES", "蓄冷罐")]), ("(d) 热泵/MW", [("K_HP", "HP", "余热热泵")])]
    sc = list(rs)
    for ax, (tt, items) in zip(axs.ravel(), groups):
        w = 0.8 / len(items); x = np.arange(len(sc))
        for k, (_, key, lab) in enumerate(items):
            v = [rs[s]["x"][key] / 1e3 for s in sc]
            ax.bar(x + (k - (len(items) - 1) / 2) * w, v, w, label=lab, color=["#3C5488", "#E64B35"][k] if len(items) > 1 else "#00A087",
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(sc); ax.set_ylabel(tt[4:]); label(ax, tt[:3])
        if len(items) > 1: ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), ncol=2)
    fig.tight_layout(); save(fig, "图5_容量配置")


# ---------------------------------------------------------------- 图6 成本构成
def fig6(site):
    f = f"{RES}/{site}_tableE.csv"
    if not os.path.exists(f): return
    E = pd.read_csv(f); E["tag"] = E["场景"].str[:2]
    items = [("其他投资/万元", "非GPU设备投资", "#8491B4"), ("购电/万元", "购电", "#3C5488"), ("碳交易/万元", "碳交易", "#00A087"),
             ("惩罚/万元", "惩罚", "#F39B7F"), ("售热/万元", "售热收益", "#E64B35")]
    fig, ax = plt.subplots(figsize=(17 * PS.CM, 7 * PS.CM)); x = np.arange(len(E))
    pos = np.zeros(len(E)); neg = np.zeros(len(E))
    for col, lab, c in items:
        v = E[col].values
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=c, edgecolor="black", linewidth=0.3, label=lab)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=c, edgecolor="black", linewidth=0.3)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.plot(x, E["能源侧年化成本/万元"], "kD", ms=4, label="能源侧年化成本")
    for i, v in enumerate(E["能源侧年化成本/万元"]): ax.text(i, pos[i] + 30, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.6); ax.set_ylim(neg.min() * 1.3, pos.max() * 1.12)
    ax.set_xticks(x); ax.set_xticklabels([NAME.get(t, t) for t in E["tag"]]); ax.set_ylabel("年化费用/万元")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "图6_成本构成")


# ---------------------------------------------------------------- 图7–10 典型日
def day_files(site):
    fs = sorted(glob.glob(f"{RES}/fig_data/{site}_*_逐时.csv"))
    return [(os.path.basename(f).split("_")[1], pd.read_csv(f), pd.read_csv(f.replace("逐时", "档位"))) for f in fs]


def fig7(site):
    days = day_files(site)
    if not days: return
    fig, axs = plt.subplots(1, len(days), figsize=(17 * PS.CM, 6.5 * PS.CM), sharey=True)
    for ax, (d, h, _), tag in zip(np.atleast_1d(axs), days, "abcd"):
        t = h["时"].values; sup = [("购电", h["购电"], "#3C5488"), ("光伏", h["光伏实用"], "#F2C14E"), ("储能放电", h["储能放电"], "#00A087")]
        b = np.zeros(len(t))
        for lab, v, c in sup: ax.bar(t, v / 1e3, 0.8, bottom=b, color=c, label=lab, edgecolor="none"); b += v / 1e3
        ax.bar(t, -h["储能充电"] / 1e3, 0.8, color="#7FC97F", label="储能充电")
        kaux = PR.params(); load_it = h["P_IT"] / 1e3
        cool = (h["冷机用电"] + h["干冷器用电"]) / 1e3; hp = h["热泵用电"] / 1e3
        ax.step(np.r_[t, 24] - 0.5, np.r_[load_it, load_it.iloc[-1]], where="post", color="black", lw=1.0, label="IT 功率")
        tot = load_it * (1 + 0.1075) + 0.3 + cool + hp
        ax.step(np.r_[t, 24] - 0.5, np.r_[tot, tot.iloc[-1]], where="post", color="#E64B35", lw=1.0, ls="--", label="园区总负荷")
        ax.set_xlim(-0.6, 23.6); ax.set_xticks(range(0, 24, 4)); ax.set_xlabel("时段/h"); label(ax, f"({tag}) {d}")
    np.atleast_1d(axs)[0].set_ylabel("功率/MW")
    h0, l0 = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h0, l0, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.06)); fig.tight_layout()
    save(fig, "图7_电功率平衡")


def fig8(site):
    days = day_files(site)
    if not days: return
    Z = PR.params()["ZONES"]
    fig, axs = plt.subplots(1, len(days), figsize=(17 * PS.CM, 6.5 * PS.CM))
    for ax, (d, h, _), tag in zip(np.atleast_1d(axs), days, "abcd"):
        t = h["时"].values; b = np.zeros(len(t))
        for lab, col, c in [("自然冷却", "自然冷却", "#4DBBD5"), ("机械制冷", "机械制冷", "#3C5488"), ("热泵取热", "热泵取热", "#E64B35"),
                            ("蓄冷放冷", "蓄冷放冷", "#00A087")]:
            ax.bar(t, h[col] / 1e3, 0.8, bottom=b, color=c, label=lab); b += h[col] / 1e3
        ax.bar(t, -h["蓄冷充冷"] / 1e3, 0.8, color="#91D1C2", label="蓄冷充冷")
        ax.set_xlim(-0.6, 23.6); ax.set_xticks(range(0, 24, 4)); ax.set_xlabel("时段/h"); ax.set_ylabel("冷/热功率/MW")
        a2 = ax.twinx(); a2.plot(t, h["液冷温度"], "k-o", ms=2.5, lw=1.0, label="液冷温度")
        for lo, hi in Z: a2.axhline(lo, color="gray", lw=0.4, ls=":")
        a2.set_ylim(28, 57); a2.set_ylabel("液冷温度/℃"); label(ax, f"({tag}) {d}")
    h0, l0 = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h0 + [plt.Line2D([], [], color="k", marker="o", ms=2.5)], l0 + ["液冷温度"], ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.06)); fig.tight_layout()
    save(fig, "图8_冷热平衡与液冷温度")


def fig9(site):
    days = day_files(site)
    if not days: return
    fig, axs = plt.subplots(1, len(days), figsize=(17 * PS.CM, 5.5 * PS.CM))
    for ax, (d, h, g), tag in zip(np.atleast_1d(axs), days, "abcd"):
        M = g[[f"档{k}" for k in range(1, 6)]].values.T
        im = ax.imshow(M, aspect="auto", cmap="YlOrRd", origin="lower", extent=[-0.5, 23.5, 0.5, 5.5])
        ax.set_yticks(range(1, 6)); ax.set_yticklabels([f"k{k}" for k in range(1, 6)]); ax.set_xticks(range(0, 24, 4))
        ax.set_xlabel("时段/h"); ax.set_ylabel("频率档位"); ax.set_title(f"({tag}) {d}", fontsize=9)
        fig.colorbar(im, ax=ax, label="工作GPU数/卡", pad=0.02)
    fig.tight_layout(); save(fig, "图9_GPU档位分布")


def fig10(site):
    days = day_files(site)
    if not days: return
    fig, axs = plt.subplots(1, len(days), figsize=(17 * PS.CM, 6 * PS.CM))
    for ax, (d, h, _), tag in zip(np.atleast_1d(axs), days, "abcd"):
        t = h["时"].values
        ax.bar(t - 0.2, h["可延期到达"] / 1e3, 0.4, color="#8491B4", label="可延期任务到达")
        ax.bar(t + 0.2, h["可延期处理"] / 1e3, 0.4, color="#E64B35", label="可延期任务处理")
        ax.plot(t, h["可延期积压"] / 1e3, "k-", lw=1.0, label="积压")
        ax.set_xlim(-0.6, 23.6); ax.set_xticks(range(0, 24, 4)); ax.set_xlabel("时段/h"); ax.set_ylabel("工作量/(10$^3$ GPU·h)")
        a2 = ax.twinx(); a2.step(np.r_[t, 24] - 0.5, np.r_[h["电价"], h["电价"].iloc[-1]], where="post", color="#00A087", lw=1.0, label="电价")
        a2.plot(t, h["光伏可用"] / h["光伏可用"].max().clip(1e-9) * 1.0, color="#F2C14E", lw=1.0, ls="--", label="光伏出力(归一化)")
        a2.set_ylim(0, 1.3); a2.set_ylabel("电价/(元/kWh)"); label(ax, f"({tag}) {d}")
    h1, l1 = np.atleast_1d(axs)[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07)); fig.tight_layout()
    save(fig, "图10_可延期任务")


# ---------------------------------------------------------------- 图11 SP/DRO/RO
def fig11(site):
    rows = []
    for setname, d, pre in (("基准典型日", RES, site), ("加入极端日", f"{RES}/ext2", f"{site}_ext")):
        for m, tg in (("SP", "SP_S6"), ("DRO", "DRO_S5"), ("RO", "RO_S7")):
            r = load(f"{d}/{pre}_{tg}.pkl"); o = f"{d}/{pre}_oos_{tg}.csv"
            if r is None: continue
            O = pd.read_csv(o) if os.path.exists(o) else None
            rows.append(dict(集合=setname, 方法=m, N=r["x"]["N"], MC=r["x"]["MC"] / 1e3, TES=r["x"]["TES"] / 1e3, HP=r["x"]["HP"] / 1e3,
                             GPU投资=r["a_full"]["N"] * r["x"]["N"] / 1e4, 非GPU投资=r["inv"] / 1e4,
                             样本外期望=(r["inv"] + 365 * O.Q.mean()) / 1e4 if O is not None else np.nan,
                             样本外总成本=(r["a_full"]["N"] * r["x"]["N"] + r["inv"] + 365 * O.Q.mean()) / 1e4 if O is not None else np.nan,
                             实时满足率=100 * (1 - O.LR.sum() / O.AR.sum()) if O is not None else np.nan,
                             缺额日概率=100 * (O.LR > 1).mean() if O is not None else np.nan))
    if len(rows) < 3: return
    T = pd.DataFrame(rows); T.to_csv(f"{RES}/{site}_SP_DRO_RO对照.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(17 * PS.CM, 5.8 * PS.CM))
    for ax, col, yl, tag in zip(axs, ["N", "样本外总成本", "缺额日概率"], ["GPU数量/卡", "样本外期望年化总成本/万元", "实时缺额日概率/%"], "abc"):
        for k, s in enumerate(T["集合"].unique()):
            t = T[T["集合"] == s]; x = np.arange(len(t)) + (k - 0.5) * 0.38
            ax.bar(x, t[col], 0.36, label=s, color=["#8491B4", "#E64B35"][k], edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(3)); ax.set_xticklabels(["SP", "DRO", "RO"]); ax.set_ylabel(yl); label(ax, f"({tag})")
        if col == "N": ax.set_ylim(T["N"].min() * 0.97, T["N"].max() * 1.02)
        if col == "样本外总成本": ax.set_ylim(np.nanmin(T[col]) * 0.97, np.nanmax(T[col]) * 1.01)
        if col == "缺额日概率": ax.set_ylim(0, np.nanmax(T[col]) * 1.2)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.1, 1.0)); fig.tight_layout(); save(fig, "图11_SP_DRO_RO对比")


# ---------------------------------------------------------------- 图12 C&CG 收敛
def fig12(site):
    ext = os.path.exists(f"{RES}/ext2/{site}_ext_DRO_S5.pkl")
    rs = {m: load(f"{RES}/ext2/{site}_ext_{tg}.pkl" if ext else f"{RES}/{site}_{tg}.pkl") for m, tg in (("DRO(S5)", "DRO_S5"), ("RO(S7)", "RO_S7"))}
    rs = {k: v for k, v in rs.items() if v is not None}
    if not rs: return
    fig, axs = plt.subplots(1, len(rs), figsize=(17 * PS.CM, 5.5 * PS.CM))
    for ax, (m, r), tag in zip(np.atleast_1d(axs), rs.items(), "ab"):
        H = pd.DataFrame(r["hist"]); k = H["it"].values
        ax.plot(k, H["UB"] / 1e4, "o-", color="#E64B35", label="上界 UB"); ax.plot(k, H["LB"] / 1e4, "s-", color="#3C5488", label="下界 LB")
        if "obj" in r: ax.axhline(r["obj"] / 1e4, color="gray", ls="--", lw=0.8, label="邻域搜索后")
        ax.set_xlabel("迭代次数"); ax.set_ylabel("能源侧年化成本/万元"); ax.set_xticks(k)
        a2 = ax.twinx(); a2.semilogy(k, H["gap"].clip(1e-5), "k^:", ms=3.5, label="相对间隙"); a2.set_ylabel("相对间隙")
        ax.set_title(f"({tag}) {m}，用时 {r['t']/60:.0f} min", fontsize=9); ax.legend(loc="center right")
    fig.tight_layout(); save(fig, "图12_CCG收敛")


# ---------------------------------------------------------------- 灵敏度
def sens(site, g, key):
    rows = []
    b = load(f"{RES}/{site}_sens_base.pkl")
    if b is not None: rows.append((PR.params()[key], b))
    for f in glob.glob(f"{RES}/{site}_sens_{g}_*.pkl"):
        rows.append((float(f.rsplit("_", 1)[1][:-4]), load(f)))
    rows.sort(key=lambda z: z[0])
    return pd.DataFrame([dict(v=v, inv=r["inv"] / 1e4, op=(r["obj"] - r["inv"]) / 1e4, obj=r["obj"] / 1e4, **{k: r["x"][k] for k in r["x"]},
                              N1=r["step1"]["N"], tot1=r["step1"]["obj_total"] / 1e4) for v, r in rows])


def fig13(site):
    import acceptance as AC
    A = AC.sens(site, "alpha"); Mx = AC.sens(site, "M")
    if A is None or len(A) < 3: return
    fig, axs = plt.subplots(1, 2, figsize=(17 * PS.CM, 6 * PS.CM))
    for ax, T, nm, fmt, tag in ((axs[0], A, "α1", "{:g}", "(a) 置信水平 α1"), (axs[1], Mx, "M", "{:.0f}", "(b) 样本量 M")):
        if T is None or len(T) < 2: ax.set_visible(False); continue
        T = T.sort_values("θ1")
        ax.plot(T["θ1"], T["能源侧成本_单独"], "o", mfc="white", color="gray", label="各点单独求解")
        ax.plot(T["θ1"], T["能源侧成本"], "s-", color="#E64B35", label="交叉评估（取最优方案）")
        for _, r in T.iterrows(): ax.annotate(f"{nm}={fmt.format(r['取值'])}", (r["θ1"], r["能源侧成本"]), textcoords="offset points", xytext=(3, -10), fontsize=7)
        ax.margins(x=0.08, y=0.15); ax.set_xlabel("1-范数模糊集半径 θ1"); ax.set_ylabel("能源侧年化成本/万元"); ax.set_title(tag, fontsize=9); ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); save(fig, "图13_置信水平与样本量")
    A.to_csv(f"{RES}/{site}_sens_alpha.csv", index=False)
    if Mx is not None: Mx.to_csv(f"{RES}/{site}_sens_M.csv", index=False)


def fig14():
    rows = []
    for s, nm in (("zhongwei", "中卫"), ("guian", "贵安"), ("shaoguan", "韶关")):
        r = load(f"{RES}/{s}_DRO_S5.pkl")
        if r is None: continue
        rows.append(dict(站点=nm, FC=r["x"]["FC"] / 1e3, MC=r["x"]["MC"] / 1e3, TES=r["x"]["TES"] / 1e3, HP=r["x"]["HP"] / 1e3, 能源侧=r["obj"] / 1e4))
    if len(rows) < 2: return
    T = pd.DataFrame(rows); fig, axs = plt.subplots(1, 2, figsize=(17 * PS.CM, 6 * PS.CM)); x = np.arange(len(T))
    for k, (c, lab, col) in enumerate([("FC", "自然冷却/MW", "#4DBBD5"), ("MC", "机械制冷/MW", "#3C5488"), ("HP", "热泵/MW", "#E64B35")]):
        axs[0].bar(x + (k - 1) * 0.26, T[c], 0.25, label=lab, color=col, edgecolor="black", linewidth=0.3)
    axs[0].set_xticks(x); axs[0].set_xticklabels(T["站点"]); axs[0].set_ylabel("容量/MW"); axs[0].legend(); label(axs[0], "(a)")
    axs[1].bar(x, T["TES"], 0.5, color="#00A087", edgecolor="black", linewidth=0.3); axs[1].set_xticks(x); axs[1].set_xticklabels(T["站点"])
    axs[1].set_ylabel("蓄冷容量/MWh"); label(axs[1], "(b)")
    fig.tight_layout(); save(fig, "图14_气候区对比")


def fig15(site):
    S = sens(site, "chip", "T_CHIP_MAX")
    if len(S) < 3: return
    fig, ax = plt.subplots(figsize=(8.5 * PS.CM, 6 * PS.CM))
    for k, lab, c, mk in (("MC", "机械制冷", "#3C5488", "o"), ("FC", "自然冷却", "#4DBBD5", "s"), ("HP", "热泵", "#E64B35", "^")):
        ax.plot(S["v"], S[k] / 1e3, marker=mk, color=c, label=lab)
    ax.set_xlabel("芯片温度上限/℃"); ax.set_ylabel("容量/MW"); ax.legend()
    a2 = ax.twinx(); a2.plot(S["v"], S["TES"] / 1e3, "k--d", ms=3.5, label="蓄冷"); a2.set_ylabel("蓄冷容量/MWh")
    save(fig, "图15_芯片温限"); S.to_csv(f"{RES}/{site}_sens_chip.csv", index=False)


def fig16(site):
    S = sens(site, "heat", "C_HEAT")
    if len(S) < 3: return
    fig, ax = plt.subplots(figsize=(8.5 * PS.CM, 6 * PS.CM))
    ax.plot(S["v"], S["HP"] / 1e3, "o-", color="#E64B35", label="热泵容量"); ax.set_xlabel("热价/(元/kWh$_{\\rm th}$)"); ax.set_ylabel("热泵容量/MW")
    a2 = ax.twinx(); a2.plot(S["v"], S["obj"], "s--", color="#3C5488", label="能源侧年化成本"); a2.set_ylabel("能源侧年化成本/万元")
    ax.axvspan(0.15, 0.20, color="gray", alpha=0.15, lw=0); ax.text(0.175, 0.05, "切换区间", ha="center", fontsize=7, transform=ax.get_xaxis_transform())
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "图16_热价灵敏度"); S.to_csv(f"{RES}/{site}_sens_heat.csv", index=False)


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"
    for f in (fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig15, fig16):
        try: f(site)
        except Exception as e: print(f.__name__, "出错：", repr(e))
    try: fig14()
    except Exception as e: print("fig14 出错：", repr(e))
