"""阶段 A3/A4：模式池上的成本—K 曲线（上界），每个结果做同时充放校验。
用法：python kmode_A4.py <池: pilot | ext> [K 列表，默认 1 2 3 4 6 12]
  pilot：现有 S5 的 12 个日模式 + 5 个常值模式（与 kmode_pilot.py 相同）
  ext  ：pilot ∪ S4、SP(S6) 与含极端日集合（ext2 下 DRO/SP/RO）结果中的日模式
结果：results/kmode/A4_{池}_K{K}.pkl、A4_{池}_summary.csv、A4_{池}_const.csv"""
import os, sys, pickle, time, numpy as np, pandas as pd
os.environ.setdefault("HIGHS_THREADS", "1")
import modes as MD, mode_lp as ML, select_master as SM
R = f"{ML.ROOT}/results"; OUT = f"{R}/kmode"; os.makedirs(OUT, exist_ok=True)
pname = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [1, 2, 3, 4, 6, 12]
ctx = ML.make_ctx("zhongwei_DRO_S5"); P = ctx["P"]; J, T, S = len(P["ZONES"]), P["T"], len(ctx["days"])
srcs = [f"{R}/zhongwei_DRO_S5.pkl"]
if pname == "ext":
    srcs += [f"{R}/zhongwei_DRO_S4.pkl", f"{R}/zhongwei_SP_S6.pkl"] + [f"{R}/ext2/zhongwei_ext_{t}.pkl" for t in ("DRO_S5", "SP_S6", "RO_S7")]
raw = MD.from_results(srcs) + [(U, f"常值{P['ZONES'][j]}") for j, U in enumerate(MD.const_modes(J, T))]
pool, idx = MD.dedup([u for u, _ in raw]); origin = {}
for (u, src), w in zip(raw, idx): origin[w] = origin[w] + "=" + src if w in origin else src     # 记录全部来源
assert all(MD.is_valid(U) for U in pool), "T3：存在非法模式"
log = lambda *a, **k: print(*a, **dict(k, flush=True))
log(f"[{pname}] 模式池：原始 {len(raw)} 个，去重后 {len(pool)} 个；来源 {[os.path.basename(p) for p in srcs]}")
pickle.dump(dict(pool=pool, origin=origin), open(f"{OUT}/A4_{pname}_pool.pkl", "wb"))

# 定温基线（5 个常值模式，含校验）
crow = []
for j in range(J):
    kc = MD.key(MD.const_modes(J, T)[j]); w = next(w for w, U in enumerate(pool) if MD.key(U) == kc)
    r = ML.lp_fixed(ctx, [pool[w]] * S, check=True, reopt_tl=300, log=log)
    crow.append(dict(温区=str(P["ZONES"][j]), 能源侧_万元=r["obj"] / 1e4, LP未校验_万元=r["obj_lp"] / 1e4, 校验方式=r["how"],
                     **{c: r["x"][c] for c in ML.ECAPS}, 残留同时充放=bool(r["simul"])))
    log(f"  定温 {P['ZONES'][j]}：{r['obj'] / 1e4:.2f} 万元（LP {r['obj_lp'] / 1e4:.2f}，{r['how']}）")
pd.DataFrame(crow).to_csv(f"{OUT}/A4_{pname}_const.csv", index=False, encoding="utf-8-sig")

# K = 1 起点：逐个单一模式解 LP（不校验，取最优者作为交替起点）
t0 = time.time(); f1 = []
for w in range(len(pool)):
    r = ML._solve_fixed(ctx, [pool[w]] * S)
    f1.append(np.inf if r is None else r["obj"])
w1 = int(np.argmin(f1)); x1 = ML._solve_fixed(ctx, [pool[w1]] * S)["x"]
log(f"  单一模式最优：模式 {w1}（{origin[w1]}）{f1[w1] / 1e4:.2f} 万元，用时 {time.time() - t0:.0f} s")
rows, sols = [], {}
for K in Ks:
    starts = [("单一模式最优", x1)] + [(f"K={k} 解", sols[k]["x"]) for k in sorted(sols)[-2:]]
    best = None
    for nm, x0 in starts:
        r = SM.select_fast(ctx, pool, K, x0, procs=int(os.environ.get("PROCS", 2)), log=log, tag=f"[{pname} K={K} 起点 {nm}]")
        if r is not None and (best is None or r["obj"] < best["obj"]): best = dict(r, start=nm)
    sols[K] = best
    pickle.dump(dict(best, pool_name=pname, modes=[pool[w] for w in best["comb"]], origin=[origin[w] for w in best["comb"]]),
                open(f"{OUT}/A4_{pname}_K{K}.pkl", "wb"))
    x = best["x"]
    rows.append(dict(K=K, 能源侧_万元=best["obj"] / 1e4, LP未校验_万元=best["obj_relaxed"] / 1e4, 模式=" ".join(map(str, best["comb"])),
                     模式来源="; ".join(origin[w] for w in best["comb"]), 分配=" ".join(map(str, best["assign"])),
                     FC=x["FC"], MC=x["MC"], TES=x["TES"], HP=x["HP"], PV=x["PV"], EB=x["EB"], PB=x["PB"],
                     残留同时充放=bool(best["simul"]), 起点=best["start"], 用时_s=best["t"]))
    log(f"[{pname}] K={K}: {best['obj'] / 1e4:.2f} 万元（LP {best['obj_relaxed'] / 1e4:.2f}），模式 {best['comb']}，分配 {best['assign']}，"
        f"FC {x['FC']:.0f} MC {x['MC']:.0f} TES {x['TES']:.0f} HP {x['HP']:.0f}，残留同时充放 {best['simul']}")
    df = pd.DataFrame(rows)
    ref = df.loc[df["K"].idxmax(), "能源侧_万元"]                 # 以已完成的最大 K（K=12 为池内全自适应）为基准
    df["VoA_%"] = (df["能源侧_万元"] / ref - 1) * 100; df["VoA基准K"] = int(df["K"].max())
    df.to_csv(f"{OUT}/A4_{pname}_summary.csv", index=False, encoding="utf-8-sig")
log(f"[{pname}] 完成")
