"""并行补跑敏感性分析中的单个参数组 (结果写入 results/sensitivity_<name>.csv, 由 run_all.merge_sens() 合并)"""
import sys, pandas as pd
import config as C, run_all as RA
name = sys.argv[1]
cases = {"R_TH": [({"R_TH": C.R_TH * 1.1}, 1.1, ["M4-DRO", "M3-SP"])],
         "BETA_LIQ": [({"BETA_LIQ": b}, b, ["M4-DRO"]) for b in (0.6, 0.8)],
         "C_W": [({"C_W": c}, c, ["M4-DRO"]) for c in (10.0, 90.0)]}[name]
D = RA.load(); rows = []
for cfg, val, models in cases:
    for v in models:
        R = RA.plan(D, v, cfg=cfg); row, _ = RA.summarize(R, D, cfg); row.update(参数=name, 取值=val); rows.append(row)
        print(f"[sens] {name}={val} {v} obj={R['obj']:.4e}", flush=True)
        pd.DataFrame(rows).to_csv(C.RES / f"sensitivity_{name}.csv", index=False)
