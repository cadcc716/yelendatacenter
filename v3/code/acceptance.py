"""
验收汇总（执行方案第二节 C1–C14，按用户修订后的判据），生成 results/<site>_验收.md 与各 csv。
依赖：S1–S7 两步法结果、单层 MILP 校验、表F、样本外测试、导出的典型日数据、灵敏度结果（缺失项标注“未运行”）。
"""
import os, sys, pickle, glob
import numpy as np, pandas as pd
import params as PR, build_op as BO, analyze as AN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"


def oos(site, tag, a_full):
    f = f"{RES}/{site}_oos_{tag}.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f); r = pickle.load(open(f"{RES}/{site}_{tag}.pkl", "rb"))
    inv = r["inv"]                                                     # 非 GPU 年化投资
    return dict(期望能源侧成本=(inv + 365 * d.Q.mean()) / 1e4, 最高日运行成本=d.Q.max() / 1e4, 缺额日概率=(d.LR > 1).mean(),
                缺额时长=int(d.LR_h.sum()), 实时满足率=100 * (1 - d.LR.sum() / d.AR.sum()), 超期量=d.O.sum(), N=r["x"]["N"])


def day_checks(site, P):
    """C10、C11、C6：读取导出的 6 月 15 日与冬季典型日逐时数据。"""
    out = {}
    for f in glob.glob(f"{RES}/fig_data/{site}_*_逐时.csv"):
        date = os.path.basename(f).split("_")[1]; h = pd.read_csv(f); g = pd.read_csv(f.replace("逐时", "档位"))
        act = g[[f"档{k}" for k in range(1, 6)]].sum(1); k1 = g["档1"] / act.where(act > 1)
        val = h["时"].isin(P["VALLEY_H"])
        out[date] = dict(k1_noon=float(k1[(h["时"] >= 12) & (h["时"] < 16)].mean()), k1_night=float(k1[h["时"] < 6].mean()),
                         valley_share=float(h.loc[val, "可延期处理"].sum() / h["可延期处理"].sum()),
                         hot_share=float((h["液冷温度"] >= 45 - 1e-6).mean()))
    return out


def sens(site, group):
    rows = []
    for f in sorted(glob.glob(f"{RES}/{site}_sens_{group}_*.pkl")):
        r = pickle.load(open(f, "rb")); v = float(f.rsplit("_", 1)[1][:-4])
        rows.append(dict(取值=v, 能源侧成本=r["obj"] / 1e4, 投资=r["inv"] / 1e4, 运行期望=(r["obj"] - r["inv"]) / 1e4, **{k: r["x"][k] for k in BO.CAPS}))
    return pd.DataFrame(rows).sort_values("取值") if rows else None


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "zhongwei"; P = PR.params()
    E = AN.table_e(site); E.to_csv(f"{RES}/{site}_tableE.csv", index=False)
    Em = AN.table_e(site, mono=True)
    e = E.set_index(E["场景"].str[:2])
    F = pd.read_csv(f"{RES}/{site}_tableF.csv").set_index("方案") if os.path.exists(f"{RES}/{site}_tableF.csv") else None
    G = {t: oos(site, t, None) for t in ("DRO_S5", "SP_S6", "RO_S7")}
    Dd = day_checks(site, P)
    L = []; ok = lambda b: "通过" if b else "未通过"
    # C1
    c1 = 1 - e.loc["S5", "能源侧年化成本/万元"] / e.loc["S1", "能源侧年化成本/万元"]
    L.append(("C1", f"S5 能源侧成本较 S1 降 {c1:.1%}（门槛 ≥5%）", ok(c1 >= 0.05)))
    # C2
    if F is not None:
        c2 = e.loc["S5", "名义PUE"] < min(F.loc["S1", "复核PUE"], F.loc["S2", "复核PUE"])
        L.append(("C2", f"S5 PUE {e.loc['S5', '名义PUE']:.3f}；S1/S2 复核 PUE {F.loc['S1', '复核PUE']:.3f}/{F.loc['S2', '复核PUE']:.3f}", ok(c2)))
    # C3
    c3 = 1 - e.loc["S2", "峰值IT功率/MW"] / e.loc["S1", "峰值IT功率/MW"]
    L.append(("C3", f"S2 峰值 IT 功率较 S1 降 {c3:.1%}（5%–10%）；储能功率 {e.loc['S1', 'P_B/MW']:.2f}→{e.loc['S2', 'P_B/MW']:.2f} MW", ok(0.05 <= c3 <= 0.10 and e.loc['S2', 'P_B/MW'] <= e.loc['S1', 'P_B/MW'])))
    # C4
    c4a = e.loc["S3", "K_MC/MW"] / max(e.loc["S2", "K_MC/MW"], 1e-9) - 1
    c4b = None if F is None else max(F.loc["S2", "最不利日缺额时长/h"], F.loc["S2", "夏季典型日名义热越限/h"])
    L.append(("C4", f"S3 机械制冷较 S2 增 {c4a:.0%}（≥10%）；S3 能源侧成本 {e.loc['S3', '能源侧年化成本/万元']:.1f} vs S2 {e.loc['S2', '能源侧年化成本/万元']:.1f} 万元；"
                    f"S2 容量入完整模型夏季热越限或缺额 {c4b} h（≥4 h）", ok(c4a >= 0.10 and (c4b or 0) >= 4)))
    # C5
    c5 = 1 - e.loc["S4", "K_MC/MW"] / e.loc["S3", "K_MC/MW"]
    L.append(("C5", f"S4 机械制冷较 S3 降 {c5:.0%}（≥15%），E_TES = {e.loc['S4', 'E_TES/MWh']:.1f} MWh；自然冷却利用小时 {e.loc['S3', '自然冷却利用小时/h']:.0f}→{e.loc['S4', '自然冷却利用小时/h']:.0f}",
              ok(c5 >= 0.15 and e.loc['S4', 'E_TES/MWh'] > 0)))
    # C6
    c6 = 1 - e.loc["S5", "碳排放/t"] / e.loc["S4", "碳排放/t"]
    win = [v for k, v in Dd.items() if not k.startswith("2023-06")]
    hs = win[0]["hot_share"] if win else np.nan
    L.append(("C6", f"S5 碳排放较 S4 降 {c6:.1%}（≥10%）；冬季典型日 45–55 ℃ 时段占比 {hs:.0%}（≥50%）；余热回收 {e.loc['S5', '余热回收/MWh']:.0f} MWh", ok(c6 >= 0.10 and hs >= 0.5)))
    # C7
    inv = {k: e.loc[k, "GPU投资/万元"] + e.loc[k, "其他投资/万元"] for k in ("S6", "S5", "S7")}
    L.append(("C7", f"投资 SP {inv['S6']:.1f} / DRO {inv['S5']:.1f} / RO {inv['S7']:.1f} 万元", ok(inv["S6"] < inv["S5"] < inv["S7"])))
    # C8
    if all(G.values()):
        cmin = min(g["期望能源侧成本"] for g in G.values())
        c8 = G["DRO_S5"]["期望能源侧成本"] <= cmin * 1.02 and G["DRO_S5"]["缺额日概率"] < 0.5 * G["SP_S6"]["缺额日概率"]
        L.append(("C8", "样本外：" + "；".join(f"{k} 期望能源侧 {g['期望能源侧成本']:.1f} 万元、缺额日概率 {g['缺额日概率']:.1%}" for k, g in G.items()), ok(c8)))
    # C9
    gaps = {k: e.loc[k, "最终gap"] for k in e.index}
    its = {k: e.loc[k, "迭代次数"] for k in e.index}
    L.append(("C9", "最终 gap（相对能源侧）：" + "，".join(f"{k} {v:.2%}/{int(its[k])}次" for k, v in gaps.items()), ok(all(v <= 1e-3 for v in gaps.values()) and all(i <= 10 for i in its.values()))))
    # C10、C11
    s = Dd.get("2023-06-15")
    if s: L.append(("C10", f"6 月 15 日 12–16 时 k1 占比 {s['k1_noon']:.0%}（<30%），0–6 时 {s['k1_night']:.0%}（>60%）", ok(s["k1_noon"] < 0.3 and s["k1_night"] > 0.6)))
    if Dd: L.append(("C11", "谷时段可延期任务处理占比：" + "，".join(f"{k} {v['valley_share']:.0%}" for k, v in Dd.items()) + "（≥40%）",
                     ok(all(v["valley_share"] >= 0.4 for v in Dd.values()))))
    # C12–C14
    A = sens(site, "alpha")
    L.append(("C12", "未运行" if A is None else "α1 灵敏度：" + "，".join(f"{r.取值}: {r.能源侧成本:.1f}" for r in A.itertuples()), "未运行" if A is None else ok(A["能源侧成本"].is_monotonic_increasing)))
    L.append(("C13", "贵安、韶关气候区对比未运行", "未运行"))
    H = sens(site, "heat")
    L.append(("C14", "未运行" if H is None else "热价灵敏度 K_HP：" + "，".join(f"{r.取值}: {r.HP:.0f} kW" for r in H.itertuples()), "待判定" if H is not None else "未运行"))
    md = ["| 编号 | 结果 | 结论 |", "| --- | --- | --- |"] + [f"| {a} | {b} | {c} |" for a, b, c in L]
    open(f"{RES}/{site}_验收.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(E.round(3).T.to_string()); print("\n".join(md))
    if len(Em):
        cmp = pd.DataFrame({"场景": E["场景"], "C&CG(邻域搜索前)": [pickle.load(open(f"{RES}/{site}_{AN.TAGS[s[:2]]}.pkl", "rb"))["obj_pre_ls"] / 1e4 for s in E["场景"]]})
        cmp = cmp.merge(pd.DataFrame({"场景": Em["场景"], "单层MILP": Em["能源侧年化成本/万元"]}), on="场景")
        cmp["相对差/%"] = 100 * (cmp["单层MILP"] - cmp["C&CG(邻域搜索前)"]) / cmp["C&CG(邻域搜索前)"]
        cmp.to_csv(f"{RES}/{site}_ccg_vs_mono.csv", index=False); print(cmp.round(3).to_string())
