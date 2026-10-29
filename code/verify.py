"""
数值验证 (缩减算例: 仅夏季代表类别, 5 个联合情景), 输出 results/verify.csv
  V1  式(7) 论文原式大M(N^max) 与 收紧大M(TIGHT_BIGM) 的最优目标是否一致
  V2  式(17)(18) LP 对偶单一 MILP 与 论文第5节"逐次加入最坏分布约束"的割平面迭代 是否一致
  V3  式(12) 互斥变量 b 取二进制 与 松弛 在储能被配置时(峰谷 ±80%) 是否一致, 并统计同时充放电时段
"""
import copy, time, numpy as np, pandas as pd
import config as C, model as Mo, run_all as RA

D = copy.deepcopy(RA.load()); d = 1
for k, v in D["scen"].items(): D["scen"][k] = v[d:d + 1]
D["n_d"] = D["n_d"][d:d + 1]; D["classes"] = D["classes"][d:d + 1]
B, A, nd = RA.blocks_of(D), RA.amb_of(D), D["n_d"]
rows = []


def solve(variant, cfg=None, price=None, n_d=None):
    t = time.time()
    M, V = Mo.build(B, variant, price=D["price"] if price is None else price, n_d=nd if n_d is None else n_d, amb=A, cfg=cfg)
    sol, st, obj, g = M.solve(tl=900)
    x, out = Mo.extract(sol, V)
    sim = sum(int(((o["Pch"] > 1e-3) & (o["Pdis"] > 1e-3)).sum()) for o in out)
    return dict(状态=st, 目标值=obj, 绝对间隙=g, 时间s=time.time() - t, N=x["N"], FC=x["FC"], MC=x["MC"], PV=x["PV"],
                EB=x["EB"], 同时充放电h=sim, 变量数=M.n, 约束数=M.nr)


import sys
ONLY = sys.argv[1] if len(sys.argv) > 1 else "all"
old = pd.read_csv(C.RES / "verify.csv") if (ONLY != "all" and (C.RES / "verify.csv").exists()) else None
for flag in ((True, False) if ONLY in ("all", "V1") else ()):
    r = solve("M3-SP", cfg={"TIGHT_BIGM": flag}); r.update(检验="V1 式(7)大M", 设置=f"TIGHT_BIGM={flag}"); rows.append(r)
    print(rows[-1], flush=True)

if ONLY in ("all", "V2"):
    r = solve("M4-DRO"); r.update(检验="V2 DRO求解", 设置="LP对偶单一MILP"); rows.append(r); print(r, flush=True)
    t = time.time()
    x, hist = Mo.solve_cutting_plane(B, D["price"], nd, A, tl=900)
    h = hist[-1]
    rows.append(dict(检验="V2 DRO求解", 设置=f"割平面迭代({len(hist)}次)", 状态="收敛" if h["违背量"] <= 1 else "未收敛",
                     目标值=h["当前解上界"], 绝对间隙=h["绝对间隙"], 时间s=time.time() - t,
                     N=x["N"], FC=x["FC"], MC=x["MC"], PV=x["PV"], EB=x["EB"]))
    pd.DataFrame(hist).to_csv(C.RES / "verify_cutting_plane_history.csv", index=False)
    print(rows[-1], flush=True)

k = np.ones(24); k[C.PEAK_H] += 0.8; k[C.VALLEY_H] -= 0.8; pr = np.roll(C.LAMBDA_E * k + C.C_TD, -C.H0)
for flag in ((False, True) if ONLY in ("all", "V3") else ()):
    # 夏季类别按 365 天计 (n_d=365) 使储能被配置, 才能检验互斥约束是否起作用
    r = solve("M4-DRO", cfg={"RELAX_B": flag}, price=pr, n_d=np.array([365]))
    r.update(检验="V3 式(12)互斥", 设置=f"峰谷±80%, n_d=365, RELAX_B={flag}")
    rows.append(r); print(r, flush=True)

R = pd.DataFrame(rows)
if old is not None: R = pd.concat([old[~old.检验.str.startswith(ONLY)], R], ignore_index=True).sort_values("检验", kind="stable")
R.to_csv(C.RES / "verify.csv", index=False); rows = R.to_dict("records")
print(pd.DataFrame(rows)[["检验", "设置", "状态", "目标值", "绝对间隙", "N", "FC", "MC", "PV", "EB", "同时充放电h", "时间s"]].to_string())
