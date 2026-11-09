"""
跨方案初值改进：嵌套热启动可能使第二步停留在上一场景的局部解。对指定场景，收集同一参数下已求得的全部容量方案，
去掉该场景不允许的设备（S3 无蓄冷与热泵，S4 无热泵），按该场景模型逐典型日精确评估（含同时充放校验），取最优者，
再从其出发做容量邻域搜索。下界取该场景已知下界中的最大值，最终间隙据此重算。原结果另存为 *_orig.pkl。
用法：python improve_cross.py <场景标签，如 DRO_S4> [邻域搜索评估次数]
"""
import os, sys, glob, pickle, shutil, numpy as np
import params as PR, build_op as BO, ccg_solve as C, local_search as LS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{ROOT}/results"
tag = sys.argv[1]; nls = int(sys.argv[2]) if len(sys.argv) > 2 else 20; site = sys.argv[3] if len(sys.argv) > 3 else "zhongwei"
f = f"{R}/{site}_{tag}.pkl"; orig = f.replace(".pkl", "_orig.pkl")
if not os.path.exists(orig): shutil.copy(f, orig)
r = pickle.load(open(orig, "rb")); scen = r["scen"]; fl = BO.flags_of(scen)
P = PR.params(); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
pool = [p for p in sorted(glob.glob(f"{R}/{site}_*.pkl")) if "_orig" not in p and "fixN" not in p]
cands, lbs = {}, [r["LB"]]
for p in pool:
    q = pickle.load(open(p, "rb"))
    if q.get("params_over", {}) not in ({}, {"C_W": 1150.0}) or abs(q["x"]["N"] - r["x"]["N"]) > 0.5: continue
    x = dict(q["x"])
    if not fl["tes"]: x["TES"] = 0.0
    if not fl["heat"]: x["HP"] = 0.0
    key = tuple(round(x[k]) for k in BO.CAPS)
    cands.setdefault(key, (os.path.basename(p), x))
    if q.get("scen") == scen and q.get("theta") is not None and np.allclose(q["theta"], r["theta"]): lbs.append(q.get("LB", -np.inf))
base = dict(r["x"])
for key, (src, x) in list(cands.items()):                      # 混合候选：本场景原方案 + 其他方案的冷侧容量
    hx = dict(base, FC=x["FC"], MC=x["MC"], TES=x["TES"] if fl["tes"] else 0.0)
    k2 = tuple(round(hx[k]) for k in BO.CAPS)
    if k2 not in cands: cands[k2] = (src + "（冷侧）+原方案", hx)
print(f"[{tag}] 候选方案 {len(cands)} 个，原能源侧 {r['obj'] / 1e4:.2f} 万元", flush=True)
best = None
for key, (src, x) in cands.items():
    ev = C.evaluate(P, D, fl, x, r["days"], r["p0"], r["theta"], None, r["a"])
    ev = C.fix_simul(P, D, fl, ev, r["days"], r["p0"], r["theta"], ev.get("int_add"), r["a"])
    print(f"  候选 {src}: {ev['obj'] / 1e4:.2f} 万元", flush=True)
    if best is None or ev["obj"] < best["obj"]: best = dict(ev, src=src)
imp, hist = LS.search(P, D, scen, r["theta"], best, r["days"], r["p0"], r["a"], log=print, worst_p=C.worst_p, max_eval=nls)
if imp is not None:
    ev = C.evaluate(P, D, fl, imp["x"], r["days"], r["p0"], r["theta"], best.get("int_add"), r["a"])
    best = dict(C.fix_simul(P, D, fl, ev, r["days"], r["p0"], r["theta"], best.get("int_add"), r["a"]), src=best["src"] + "+邻域搜索")
out = dict(r)
if best["obj"] < r["obj"] - 1.0:
    out.update({k: best[k] for k in ("x", "Q", "R", "p", "inv", "obj", "simul", "int_add") if k in best})
out["LB"] = max(lbs); out["gap_final"] = (out["obj"] - out["LB"]) / abs(out["obj"])
out["improve_cross"] = dict(src=best["src"], obj_before=r["obj"], obj_after=out["obj"], n_cand=len(cands), ls_hist=hist)
pickle.dump(out, open(f, "wb"))
print(f"[{tag}] 能源侧 {r['obj'] / 1e4:.2f} → {out['obj'] / 1e4:.2f} 万元（来源 {best['src']}），下界 {out['LB'] / 1e4:.2f}，间隙 {out['gap_final']:.4f}", flush=True)
