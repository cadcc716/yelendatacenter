"""M3 容量 + 自然冷却余量的统一评价 (长负荷数据): python fc_margin.py <FC kWth>"""
import pickle, sys, numpy as np, pandas as pd, exp as X, risk_e as RK
fc = float(sys.argv[1]); D = X.dataset("long"); R = pickle.load(open("../results/exp/plan_L-M3.pkl", "rb"))
x = dict(R["x"]); x["FC"] = fc
E = X.evaluate(x, D, cfg=R["cfg"]); E.to_csv(f"../results/exp/hf_L-M3-FC{int(fc)}.csv", index=False)
R2 = dict(R, x=x); s = X.hf_summary(E, R2, D); nd = D["n_d"]
sR = RK.annual(E, E["实时失约"].fillna(0).values, nd); gl = RK.annual(E, E["实时失约_GPU台数受限"].fillna(0).values, nd); sD = RK.annual(E, E["延期失约"].fillna(0).values, nd)
print(f"[fcm] FC={fc:.0f} 能源相关费用={s['复核能源相关费用']/1e4:.1f} 冷却受限失约={sR-gl:.0f} 风险调整={ (s['复核能源相关费用'] + 100*sR + 20*sD)/1e4:.1f} SLA={s['实时SLA']*100:.2f}", flush=True)
