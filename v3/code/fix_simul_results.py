"""对已保存结果中存在同时充放（邻域搜索评估时 0-1 变量松弛所致）的方案，固定容量、加回相应 0-1 变量重新逐日求解，更新运行结果与目标值。"""
import os, sys, glob, pickle, numpy as np
import params as PR, build_op as BO, ccg_solve as C
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sys.argv[1:] or sorted(glob.glob(f"{ROOT}/results/*.pkl") + glob.glob(f"{ROOT}/results/ext2/*.pkl"))
for f in files:
    r = pickle.load(open(f, "rb"))
    if not r.get("simul"): continue
    site = "zhongwei_ext" if "_ext_" in os.path.basename(f) else os.path.basename(f).split("_")[0]
    P = PR.params(**r.get("params_over", {})); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    flags = BO.flags_of(r["scen"]); ev = dict(x=r["x"], R=r["R"])
    old = r["obj"]
    ev = C.fix_simul(P, D, flags, ev, r["days"], r["p0"], r["theta"], r.get("int_add"), r["a"])
    r.update({k: ev[k] for k in ("Q", "R", "p", "inv", "obj", "simul", "int_add") if k in ev})
    r["gap_final"] = (r["obj"] - r["LB"]) / abs(r["obj"])
    pickle.dump(r, open(f, "wb"))
    print(f"{os.path.basename(f)}: 能源侧 {old / 1e4:.2f} → {r['obj'] / 1e4:.2f} 万元，剩余同时充放 {r['simul'] or '无'}，gap {r['gap_final']:.4f}", flush=True)
