"""审稿式物理一致性检查：读取导出的逐时数据，检验功率平衡、SOC/蓄冷边界、同时充放、电网上限与价格响应。"""
import sys, glob, os, pickle, numpy as np, pandas as pd
import params as PR
P = PR.params(); kaux = P["DELTA_E"] + P["A_CIRC"] * P["BETA"] + P["A_AIR"] * (1 - P["BETA"])
pr = PR.price(P)
for d in (sys.argv[1:] or ["../results/fig_data", "../results/fig_data_DRO_S1", "../results/fig_data_DRO_S3", "../results/fig_data_DRO_S4"]):
    tag = os.path.basename(d.rstrip("/"))
    src = {"fig_data": "zhongwei_DRO_S5", "fig_data_DRO_S1": "zhongwei_DRO_S1", "fig_data_DRO_S3": "zhongwei_DRO_S3", "fig_data_DRO_S4": "zhongwei_DRO_S4"}[tag]
    x = pickle.load(open(f"../results/{src}.pkl", "rb"))["x"]
    for f in sorted(glob.glob(f"{d}/*_逐时.csv")):
        h = pd.read_csv(f); date = os.path.basename(f).split("_")[1]
        res = h["购电"] + h["光伏实用"] + h["储能放电"] - (1 + kaux) * h["P_IT"] - h["干冷器用电"] - h["冷机用电"] - h["热泵用电"] - h["储能充电"] - P["P0"]
        soc = h["储能SOC"] / max(x["EB"], 1e-9)
        tes = h["蓄冷库存"] / max(x["TES"], 1e-9) if x["TES"] > 1 else pd.Series([0.0])
        simB = ((h["储能充电"] > 1) & (h["储能放电"] > 1)).sum(); simT = ((h["蓄冷充冷"] > 1) & (h["蓄冷放冷"] > 1)).sum()
        pv_ok = (h["光伏实用"] <= h["光伏可用"] + 1e-3).all()
        ch_val = h.loc[h["时"].isin(P["VALLEY_H"]), "储能充电"].sum() / max(h["储能充电"].sum(), 1e-9)
        dis_peak = h.loc[h["时"].isin(P["PEAK_H"]), "储能放电"].sum() / max(h["储能放电"].sum(), 1e-9)
        heat_ok = (h["热泵供热"] <= h["热需求"] + 1e-3).all()
        print(f"{tag:18s} {date}  功率平衡最大残差 {res.abs().max():.3f} kW | SOC {soc.min():.2f}–{soc.max():.2f} | 蓄冷 {tes.min():.2f}–{tes.max():.2f} | "
              f"同时充放 储能{simB}/蓄冷{simT} h | 光伏≤可用 {pv_ok} | 弃光 {h['弃光'].sum():.0f} kWh | 购电峰值 {h['购电'].max():.0f} kW | "
              f"储能充电在谷段 {ch_val:.0%}、放电在峰段 {dis_peak:.0%} | 供热≤需求 {heat_ok} | 实时切除 {h['实时切除'].sum():.0f} | 超期 {h['超期量'].sum():.0f}")
