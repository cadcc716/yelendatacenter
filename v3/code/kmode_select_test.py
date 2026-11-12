"""A5 核对：模式选址主问题 MILP（select_master.select）以交替法的 K=2 解热启动，检查上界不劣于热启动值。"""
import os, sys, pickle, time, numpy as np
os.environ.setdefault("HIGHS_THREADS", "1")
import mode_lp as ML, select_master as SM
OUT = f"{ML.ROOT}/results/kmode"; pname = sys.argv[1] if len(sys.argv) > 1 else "pilot"; K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ctx = ML.make_ctx("zhongwei_DRO_S5"); pool = pickle.load(open(f"{OUT}/A4_{pname}_pool.pkl", "rb"))["pool"]
st = pickle.load(open(f"{OUT}/A4_{pname}_K{K}.pkl", "rb"))
t = time.time(); mask = SM.feasible_mask(ctx, pool, procs=1); print(f"可行性剔除：{(~mask).sum()} / {mask.size} 个 (日, 模式) 组合不可行，{time.time() - t:.0f} s", flush=True)
r = SM.select(ctx, pool, K, start=dict(assign=st["assign"]), mask=mask, tl=float(os.environ.get("TL", 600)), int_add=st.get("int_add"))
print(f"热启动（交替法，校验后）{st['obj'] / 1e4:.2f} 万元；主问题 MILP 上界 {r['obj'] / 1e4:.2f} 万元，下界 {r['LB'] / 1e4:.2f} 万元，"
      f"状态 {r['status']}，模式 {r.get('comb')}，用时 {r['t']:.0f} s", flush=True)
pickle.dump(r, open(f"{OUT}/A5_select_{pname}_K{K}.pkl", "wb"))
