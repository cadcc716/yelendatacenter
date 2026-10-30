"""联合候选补测: 对未收敛方案试算若干由相关方案得到的容量候选 (固定容量下同一模型的完整目标), 若更优则替换并重做复核。"""
import pickle, sys, numpy as np, exp as X
RES = X.RES
cands = {
    "S1-aN0.6-M3": ["M3"], "S1-aN1.4-M3": ["M3"],
    "S2-坦帕-M3": [dict(MC=229.0, FC=4129.0), dict(MC=0.0, FC=5000.0), dict(MC=500.0, FC=3500.0), dict(MC=800.0, FC=3000.0)],
    "S2-芝加哥-M3": ["S2-芝加哥-M0"],
}
name = sys.argv[1]
R = pickle.load(open(RES / f"plan_{name}.pkl", "rb")); lb = R["obj"] - R["gap"]
best, fb = dict(R["x"]), R["obj"]; ob = R["out"]
for c in cands[name]:
    x = dict(R["x"])
    if isinstance(c, str): src = pickle.load(open(RES / f"plan_{c}.pkl", "rb"))["x"]; x.update({k: src[k] for k in ("PV", "FC", "MC", "PB", "EB")})
    else: x.update(c)
    fx, ox = X.plan_cost(name, x); print(f"[seed] {name} 候选 {dict((k, round(v)) for k, v in x.items())} → {fx:.6e}", flush=True)
    if fx < fb - 1.0: best, fb, ob = x, fx, ox
if fb < R["obj"] - 1.0:
    pickle.dump(dict(R, x=best, out=ob, obj=fb, gap=fb - lb, status=R["status"].split("+")[0] + "+联合候选"), open(RES / f"plan_{name}.pkl", "wb"))
    R2 = pickle.load(open(RES / f"plan_{name}.pkl", "rb")); D = X.dataset(X.PLANS[name].get("ds", "base"))
    E = X.evaluate(R2["x"], D, cfg=R2["cfg"]); E.insert(0, "方案", name); E.to_csv(RES / f"hf_{name}.csv", index=False)
    row = pickle.load(open(RES / f"row_{name}.pkl", "rb"))
    row.update(状态=R2["status"], 间隙=R2["gap"], 目标值=fb, **best, **{f"年化_{k}": v for k, v in X.capex(R2).items()})
    row["年化投资"] = sum(X.capex(R2).values())
    row["规划期望运行费用"] = float(sum(b["p"] * R2["n_d"][b["d"]] * o["Cop"] for b, o in zip(R2["blocks"], R2["out"])))
    row["规划能源相关费用"] = row["年化投资"] - row["年化_N"] + row["规划期望运行费用"]
    row.update(X.hf_summary(E, R2, D)); pickle.dump(row, open(RES / f"row_{name}.pkl", "wb"))
print(f"[seed] {name} 完成: 目标 {fb:.6e} 间隙 {fb - lb:.0f} x=" + " ".join(f"{k}:{v:.0f}" for k, v in best.items()), flush=True)
