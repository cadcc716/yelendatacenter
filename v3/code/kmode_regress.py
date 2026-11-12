"""回归测试：不传新参数时 add_day 行为不变——用现有 S5 容量逐日求解，与 zhongwei_DRO_S5.pkl 的 Q 对比。"""
import os, pickle, numpy as np
os.environ.setdefault("HIGHS_THREADS", "1")
import params as PR, build_op as BO, ccg_solve as C
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = pickle.load(open(f"{ROOT}/results/zhongwei_DRO_S5.pkl", "rb"))
P = PR.params(**(r.get("params_over") or {})); D = dict(np.load(f"{ROOT}/data/zhongwei_data.npz"))
ev = C.evaluate(P, D, BO.flags_of("S5"), r["x"], r["days"], r["p0"], r["theta"], r.get("int_add"), r["a"])
d = np.abs(ev["Q"] - np.asarray(r["Q"])); rel = d / np.abs(r["Q"])
for s, (q0, q1) in enumerate(zip(r["Q"], ev["Q"])): print(f"日 {s:2d}: pkl {q0:12.2f}  重算 {q1:12.2f}  差 {q1 - q0:+.2f} 元")
print(f"最大相对差 {rel.max():.2e}；目标 pkl {r['obj'] / 1e4:.4f} 万元，重算 {ev['obj'] / 1e4:.4f} 万元")
