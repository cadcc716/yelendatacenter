"""
修订版实验方案 E1–E5 主程序
  E1 物理机制两因素消融 M0–M3 (相同经验概率, 只切换 a=GPU调节, b=热状态/气象冷却反馈) + M0-W45 静态温水设计
  E2 容量互补: 容量差值、活跃约束统计、统一物理复核 (同一高保真评价器, 字典序失约诊断)、交互项、修复成本、同容量运行比较
  E3 不确定性: U0 确定性均值 / U-SP(=M3) / U1 无条件对称 / U2 条件对称 / U3 条件正负半方差 (本文); κ 前沿; 连续压力测试
  E4 敏感性: S1 GPU 年化投资; S2 气候 (自然冷却有效窗口); S3 可延期任务比例; 温区网格精度
  E5 直接对偶 MILP vs 概率割平面
用法: python exp.py plan <方案>   |  python exp.py ops  |  python exp.py bench <类数> <算法>  |  python exp.py collect
"""
import sys, time, pickle, copy
import numpy as np, pandas as pd
import config as C, model as Mo

RES = C.RES / "exp"; RES.mkdir(parents=True, exist_ok=True)
HF = (True, "dyn", "SP")          # 统一高保真评价器: 完整 M3 物理与运行控制
CLIMATE = {"旧金山": "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw", "芝加哥": "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
           "坦帕": "USA_FL_Tampa.Intl.AP.722110_TMY3.epw"}


def cost_scaled(f):
    c = dict(C.COST); v = c["N"]; c["N"] = (v[0] * f, v[1], v[2] * f); return c


ZONES_FINE = [(25.0, 28.5), (28.5, 32.0), (32.0, 35.0), (35.0, 38.0), (38.0, 41.5), (41.5, 45.0),
              (45.0, 47.5), (47.5, 50.0), (50.0, 51.5), (51.5, 53.0)]

PLANS = {  # 名称: spec=(a, 冷却模型, 概率层), sup=支持集, ds=数据集, cfg=参数覆盖, full=是否做压力测试与修复诊断
    "M0": dict(spec=(False, "static", "SP"), full=True),
    "M1": dict(spec=(True, "static", "SP"), full=True),
    "M2": dict(spec=(False, "dyn", "SP"), full=True),
    "M3": dict(spec=(True, "dyn", "SP"), full=True),                       # = U-SP
    "M0-W45": dict(spec=(False, "static", "SP"), cfg={"TW_STATIC": 45.0}, full=True),
    "U0": dict(spec=(True, "dyn", "SP"), sup="U0", full=True),
    "U1": dict(spec=(True, "dyn", "DRO"), sup="U1", full=True),
    "U2": dict(spec=(True, "dyn", "DRO"), amb="U2", full=True),
    "U3": dict(spec=(True, "dyn", "DRO"), amb="U3", full=True),            # 本文方法
}
for k in (0.5, 2.0):
    for u in ("U1", "U2", "U3"):
        PLANS[f"{u}-k{k:g}"] = dict(PLANS[u], ds=f"k{k:g}", full=False)
for f in (0.6, 1.4):
    for m in ("M0", "M3"):
        PLANS[f"S1-aN{f:g}-{m}"] = dict(PLANS[m], cfg={"COST": cost_scaled(f)}, full=False)
for c in CLIMATE:
    for m in ("M0", "M3"):
        PLANS[f"S2-{c}-{m}"] = dict(PLANS[m], ds=c, full=False)
for sh in ("b15", "b50"):
    for m in ("M0", "M3"):
        PLANS[f"S3-{sh}-{m}"] = dict(PLANS[m], ds=sh, full=False)
PLANS["温区加密-M3"] = dict(PLANS["M3"], cfg={"ZONES": ZONES_FINE}, full=False)

_DS = {}


def dataset(tag="base"):
    if tag not in _DS:
        f = C.PROC / f"dataset_{tag}.pkl"
        if not f.exists():
            import data_e as DE
            kw = {}
            if tag in CLIMATE: kw["epw"] = CLIMATE[tag]
            if tag.startswith("k"): kw["kappa"] = float(tag[1:])
            if tag in ("b15", "b50"):
                tot, sh = C.R_MEAN_GPU + C.B_MEAN_GPU, int(tag[1:]) / 100
                kw.update(r_mean=tot * (1 - sh), b_mean=tot * sh)
            pickle.dump(DE.build(**kw), open(f, "wb"))
        _DS[tag] = pickle.load(open(f, "rb"))
    return _DS[tag]


def support(D, pl):
    """返回 (blocks, n_d, amb): U0 均值轨迹; U1 全年合并 (单一类别, n=365); 其余为季节条件支持集。"""
    tb = lambda b: dict(b, Tstat=D["Tbar"][b["d"]])
    if pl.get("sup") == "U0":
        return [tb(b) for b in D["U0"]], D["n_d"], None
    blocks = [tb(b) for S in D["sup"] for b in S]
    if pl.get("sup") == "U1":
        nd = D["n_d"]; out = []
        for i, b in enumerate(blocks):
            out.append(dict(b, d=0, s=i, p=b["p"] * nd[b["d"]] / nd.sum(), d0=b["d"]))
        return out, np.array([float(nd.sum())]), [D["amb_U1"]]
    return blocks, D["n_d"], (D["amb"][pl["amb"]] if "amb" in pl else None)


def plan(name, tl=C.TIME_LIMIT):
    pl = PLANS[name]; D = dataset(pl.get("ds", "base")); cfg = pl.get("cfg")
    B, nd, amb = support(D, pl)
    t0 = time.time()
    M, V = Mo.build(B, pl["spec"], price=D["price"], n_d=nd, amb=amb, cfg=cfg)
    sol, st, obj, g = M.solve(tl=tl)
    if sol is None: raise RuntimeError(f"{name}: {st}")
    x, out = Mo.extract(sol, V)
    if pl["spec"][2] != "SP":   # 退化解: 固定 x 按式(32) Q_ds(x) 逐情景重解
        for i, b in enumerate(B):
            Mb, Vb = Mo.build([dict(b, d=0)], pl["spec"][:2] + ("SP",), x_fix=x, price=D["price"], n_d=np.ones(1), cfg=cfg)
            sb = Mb.solve(gap=1e-6, tl=120)[0]; o = Mo.extract(sb, Vb)[1][0]; o.update(d=b["d"], s=b["s"]); out[i] = o
    pw = None
    if amb is not None:
        pw = []
        for d in range(len(nd)):
            Q = np.array([o["Cop"] for o in out if o["d"] == d]); pw.append(Mo.worst_case_p(Q, amb[d]))
    return dict(name=name, x=x, out=out, obj=obj, gap=g, status=st, time=time.time() - t0, a=V["a"], nvar=M.n, nrow=M.nr,
                blocks=B, n_d=nd, pw=pw, cfg=cfg)


def capex(R):
    return {k: R["a"][k] * R["x"][k] for k in ["N", "PV", "PB", "EB", "FC", "MC"]}


def eval_day(x, b, price, spec=HF, cfg=None, tl=90):
    M, V = Mo.build([dict(b, d=0, s=0, p=1.0)], spec, x_fix=x, slack=True, price=price, n_d=np.ones(1), cfg=cfg)
    sol, st, obj, g = M.solve(gap=0.002, tl=tl)
    if sol is None:   # 即使允许任务失约也无法满足热安全/功率平衡: 记为"不可安全运行" (修订版 5.3.4)
        return dict(状态="不可安全运行", 不可安全运行=1, 日运行费用=np.nan, 实时失约=np.nan, 延期失约=np.nan, 实时失约_GPU台数受限=np.nan,
                    实时到达=b["AR"].sum(), 延期到达=b["AB"].sum(), Tmax=b["Tamb"].max()), None
    o = Mo.extract(sol, V)[1][0]
    gpu_lim = float(np.clip(b["AR"] - x["N"], 0, None).sum())          # 即使全部 GPU 标称满速也无法服务的实时工作量
    return dict(状态=st, 日运行费用=o["Cop"], 实时失约=o["sR"].sum(), 延期失约=o["sD"].sum(), 实时失约_GPU台数受限=min(gpu_lim, o["sR"].sum()),
                实时到达=b["AR"].sum(), 延期到达=b["AB"].sum(), IT电量=o["PIT"].sum(), 冷却电量=(o["PFC"] + o["PMC"]).sum(),
                机械制冷电量=o["PMC"].sum(), 购电量=o["Pg"].sum(), 购电峰值=o["Pg"].max(), 光伏利用=o["PPV"].sum(),
                光伏可用=o["PVav"].sum(), 液冷温度max=o["Tw"].max(), Tmax=b["Tamb"].max(), 总电量=o["Ptot"].sum(), 不可安全运行=0), o


def evaluate(x, D, spec=HF, cfg=None, days="test"):
    rows = []
    for d, S in enumerate(D[days]):
        for i, b in enumerate(S):
            r, _ = eval_day(x, b, D["price"], spec, cfg); r.update(d=d, i=i, w=b["w"], l=b["l"]); rows.append(r)
    return pd.DataFrame(rows)


def hf_summary(E, R, D):
    nd = D["n_d"]; per = E.groupby("d").mean(numeric_only=True)
    ann = lambda c: float((nd * per[c].reindex(range(4)).values).sum())
    cx = capex(R); ce = sum(cx.values())
    c = E["日运行费用"].values; eta = np.quantile(c, 0.95)
    fail = ((E["实时失约"] + E["延期失约"]) > 1e-3) | (E.get("不可安全运行", 0) == 1)
    out = dict(不可安全运行日数=int(E.get("不可安全运行", pd.Series([0])).sum()), 复核年运行费用=ann("日运行费用"), 复核年化总费用=ce + ann("日运行费用"),
               复核能源相关费用=ce - cx["N"] + ann("日运行费用"),
               实时SLA=1 - E["实时失约"].sum() / E["实时到达"].sum(), 延期SLA=1 - E["延期失约"].sum() / E["延期到达"].sum(),
               实时失约GPUh=E["实时失约"].sum(), 延期失约GPUh=E["延期失约"].sum(),
               实时失约_GPU台数受限=E["实时失约_GPU台数受限"].sum(),
               实时失约_冷却热安全受限=E["实时失约"].sum() - E["实时失约_GPU台数受限"].sum(),
               失约日比例=float(fail.mean()), 日费用CVaR95=float(eta + np.clip(c - eta, 0, None).mean() / 0.05),
               复核冷却电量MWh=ann("冷却电量") / 1e3, 复核购电量MWh=ann("购电量") / 1e3, 复核PUE=ann("总电量") / ann("IT电量"),
               复核光伏利用率=E["光伏利用"].sum() / max(E["光伏可用"].sum(), 1e-9), 购电峰值kW=E["购电峰值"].max())
    return out


def stress(R, D):
    rows = []
    for k, s in D["stress"].items():
        T = len(s["AR"]); cfg = dict(R["cfg"] or {}); cfg["T"] = T
        b = dict(AR=s["AR"], AB=s["AB"], Tamb=s["Tamb"], gpv=s["gpv"])
        M, V = Mo.build([dict(b, d=0, s=0, p=1.0)], HF, x_fix=R["x"], slack=True, price=np.tile(D["price"], T // 24),
                        n_d=np.ones(1), cfg=cfg)
        sol, st, obj, g = M.solve(gap=0.002, tl=600)
        o = Mo.extract(sol, V)[1][0]
        P = C.params(cfg)
        margin = []
        for kk, m in o["modes"].items():
            h = P["MODES"][kk.split("|")[1]][1]; act = m > 1e-6
            if act.any(): margin.append(P["T_CHIP_MAX"] - (np.maximum(o["Tw"][:-1], o["Tw"][1:])[act].max() + P["R_TH"] * h))
        rows.append(dict(方案=R["name"], 压力片段=k, 状态=st, 运行费用=o["Cop"], 实时失约=o["sR"].sum(), 延期失约=o["sD"].sum(),
                         最小热安全裕量K=min(margin) if margin else np.nan, 液冷温度max=o["Tw"].max(), 机械制冷电量=o["PMC"].sum(),
                         SOC_min=float(o["S"].min()), Tmax=s["Tamb"].max()))
        pickle.dump(o, open(RES / f"stress_{R['name']}_{k.split()[0]}.pkl", "wb"))
    return pd.DataFrame(rows)


def repair(R, D, E, tl=900):
    """5.3.4 事后容量修复诊断: 在安装上限内以最小新增年化投资使全部失约测试日可安全运行 (仅用于评价)。"""
    bad = E[(E["实时失约"] + E["延期失约"]) > 1e-3]
    if bad.empty: return dict(修复日数=0, 修复成本=0.0, **{f"修复_{k}": 0.0 for k in ["N", "PV", "PB", "EB", "FC", "MC"]})
    B = [dict(D["test"][int(r.d)][int(r.i)], d=0, s=j, p=1.0) for j, r in enumerate(bad.itertuples())]
    M, V = Mo.build(B, HF, x_lb=R["x"], price=D["price"], n_d=np.array([1e-3]), cfg=R["cfg"])
    sol, st, obj, g = M.solve(tl=tl)
    if sol is None: return dict(修复日数=len(B), 修复状态=st)
    x2 = Mo.extract(sol, V)[0]
    dx = {k: max(0.0, x2[k] - R["x"][k]) for k in x2}
    return dict(修复日数=len(B), 修复状态=st, 修复成本=sum(R["a"][k] * dx[k] for k in dx), **{f"修复_{k}": v for k, v in dx.items()})


def repair_cool(R, D, E, tl=900):
    """冷却/热安全修复: GPU 台数固定为规划值, 实时到达量截断到 N (GPU 台数不足部分各方案共有, 单列),
    以最小新增冷源/储能/光伏年化投资使失约测试日在完整物理约束下可安全运行。"""
    N = R["x"]["N"]
    bad = E[(E["实时失约"] - E["实时失约_GPU台数受限"] + E["延期失约"]) > 1e-3]
    if bad.empty: return dict(冷却修复日数=0, 冷却修复成本=0.0, 冷却修复_FC=0.0, 冷却修复_MC=0.0)
    B = []
    for j, r in enumerate(bad.itertuples()):
        b = dict(D["test"][int(r.d)][int(r.i)]); b["AR"] = np.minimum(b["AR"], N * 0.999); B.append(dict(b, d=0, s=j, p=1.0))
    cfg = dict(R["cfg"] or {}); X = dict(C.XMAX); X["N"] = int(round(N)); cfg["XMAX"] = X
    M, V = Mo.build(B, HF, x_lb=R["x"], price=D["price"], n_d=np.array([1e-3]), cfg=cfg)
    sol, st, obj, g = M.solve(tl=tl)
    if sol is None: return dict(冷却修复日数=len(B), 冷却修复状态=st)
    x2 = Mo.extract(sol, V)[0]; dx = {k: max(0.0, x2[k] - R["x"][k]) for k in x2}
    return dict(冷却修复日数=len(B), 冷却修复状态=st, 冷却修复成本=sum(R["a"][k] * dx[k] for k in dx),
                **{f"冷却修复_{k}": dx[k] for k in ("FC", "MC", "PV", "PB", "EB")})


def activity(R):
    """E2 活跃约束统计 (规划情景, 经验概率加权年小时数)。"""
    P = C.params(R["cfg"]); w = []
    for b, o in zip(R["blocks"], R["out"]):
        w.append(b["p"] * R["n_d"][b["d"]])
    w = np.array(w); out = {}
    zones = P["ZONES"]
    for j in range(len(zones)):
        out[f"温区{j}小时"] = float(sum(wi * (o["zone"][:-1] == j).sum() for wi, o in zip(w, R["out"])))
    out["机械制冷运行小时"] = float(sum(wi * (o["qMC"] > 1e-3).sum() for wi, o in zip(w, R["out"])))
    md = {}
    for wi, o in zip(w, R["out"]):
        for k, v in o["modes"].items(): md[k] = md.get(k, 0) + wi * v.sum()
    tot = sum(md.values())
    out.update({f"模式占比_{k}": v / tot for k, v in md.items()})
    return out


def job(name):
    import os
    R = plan(name, tl=float(os.environ.get("TL", C.TIME_LIMIT)))
    pickle.dump(R, open(RES / f"plan_{name}.pkl", "wb"))
    print(f"[plan] {name} {R['status']} obj={R['obj']:.5e} gap={R['gap']:.0f} t={R['time']:.0f}s x=" +
          " ".join(f"{k}:{v:.0f}" for k, v in R["x"].items()), flush=True)
    D = dataset(PLANS[name].get("ds", "base"))
    E = evaluate(R["x"], D, cfg=R["cfg"]); E.insert(0, "方案", name); E.to_csv(RES / f"hf_{name}.csv", index=False)
    row = dict(方案=name, 状态=R["status"], 间隙=R["gap"], 求解时间s=R["time"], 变量数=R["nvar"], 约束数=R["nrow"], 目标值=R["obj"],
               **{k: v for k, v in R["x"].items()}, **{f"年化_{k}": v for k, v in capex(R).items()})
    row["年化投资"] = sum(capex(R).values())
    nd = R["n_d"]
    row["规划期望运行费用"] = float(sum(b["p"] * nd[b["d"]] * o["Cop"] for b, o in zip(R["blocks"], R["out"])))
    row["规划能源相关费用"] = row["年化投资"] - row["年化_N"] + row["规划期望运行费用"]
    row.update(hf_summary(E, R, D)); row.update(activity(R))
    if PLANS[name]["full"]:
        S = stress(R, D); S.to_csv(RES / f"stress_{name}.csv", index=False)
        row.update(repair(R, D, E)); row.update(repair_cool(R, D, E))
    pickle.dump(row, open(RES / f"row_{name}.pkl", "wb"))
    print(f"[done] {name} 复核年化总费用={row['复核年化总费用']:.5e} 实时SLA={row['实时SLA']:.5f} 修复={row.get('修复成本', '-')}", flush=True)


def ops():
    """5.3.3 同容量运行比较: 固定 x_ref = M3 规划容量, 只改变运行策略 (a, b)。"""
    R = pickle.load(open(RES / "plan_M3.pkl", "rb")); D = dataset()
    rows = []
    for nm, spec in {"M0策略": (False, "fixed", "SP"), "M1策略": (True, "fixed", "SP"), "M2策略": (False, "dyn", "SP"),
                     "M3策略": (True, "dyn", "SP")}.items():
        E = evaluate(R["x"], D, spec=spec); E.insert(0, "策略", nm); E.to_csv(RES / f"ops_{nm}.csv", index=False)
        s = hf_summary(E, R, D); s["策略"] = nm; rows.append(s); print("[ops]", nm, s["复核年运行费用"], s["实时SLA"], flush=True)
    pd.DataFrame(rows).to_csv(RES / "ops_summary.csv", index=False)


def bench(nc, alg):
    """E5: 相同 U3 模型, 规模 = 前 nc 个季节类别; A0 对偶单一 MILP / A1 概率割平面。"""
    nc = int(nc); D = dataset(); pl = PLANS["U3"]
    B, nd, amb = support(D, pl); B = [b for b in B if b["d"] < nc]; nd = nd[:nc]; amb = amb[:nc]
    t0 = time.time()
    if alg == "A0":
        M, V = Mo.build(B, pl["spec"], price=D["price"], n_d=nd, amb=amb)
        sol, st, obj, g = M.solve(tl=1800); x = Mo.extract(sol, V)[0]; it = 1; hist = None
        r = dict(规模=nc, 算法="A0 对偶单一MILP", 状态=st, 目标值=obj, 绝对间隙=g, 迭代=1, 变量数=M.n, 约束数=M.nr)
    else:
        x, hist = Mo.solve_cutting_plane(B, D["price"], nd, amb, tl=1800)
        h = hist[-1]; r = dict(规模=nc, 算法="A1 概率割平面", 状态="收敛" if h["违背量"] <= 1 else "未收敛", 目标值=h["当前解上界"],
                               绝对间隙=h["绝对间隙"], 迭代=len(hist))
        pd.DataFrame(hist).to_csv(RES / f"bench_hist_{nc}.csv", index=False)
    r.update(时间s=time.time() - t0, 情景数=len(B), **{k: v for k, v in x.items()})
    pd.DataFrame([r]).to_csv(RES / f"bench_{nc}_{alg}.csv", index=False); print("[bench]", r, flush=True)


def design_check():
    """设计极端日校核 (ASHRAE n=20 年一遇): 各方案容量在设计日上的统一物理评价 (设计日来自历史资料, 不属于测试集)。"""
    rows = []
    for f in sorted(RES.glob("plan_*.pkl")):
        R = pickle.load(open(f, "rb")); pl = PLANS.get(R["name"])
        if pl is None: continue
        D = dataset(pl.get("ds", "base")); b = [x for S in D["sup"] for x in S if x["tag"].startswith("设计极端日")][0]
        r, o = eval_day(R["x"], b, D["price"], cfg=R["cfg"], tl=300)
        rows.append(dict(方案=R["name"], 设计温度=D["design"]["T"], 状态=r["状态"], 实时失约=r.get("实时失约"), 延期失约=r.get("延期失约"),
                         实时失约_GPU台数受限=r.get("实时失约_GPU台数受限"), 机械制冷电量=r.get("机械制冷电量"), 液冷温度max=r.get("液冷温度max")))
        print("[design]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "design_check.csv", index=False)


def collect():
    rows = [pickle.load(open(f, "rb")) for f in sorted(RES.glob("row_*.pkl"))]
    S = pd.DataFrame(rows).set_index("方案")
    S.to_csv(RES / "summary.csv")
    st = sorted(f for f in RES.glob("stress_*.csv") if f.name != "stress_all.csv"); hf = sorted(RES.glob("hf_*.csv")); bn = sorted(RES.glob("bench_*_A*.csv"))
    if st: pd.concat([pd.read_csv(f) for f in st]).to_csv(RES / "stress_all.csv", index=False)
    if bn: pd.concat([pd.read_csv(f) for f in bn]).to_csv(RES / "bench_all.csv", index=False)
    D = dataset(); pd.DataFrame(D["calib"]).to_csv(RES / "calibration.csv", index=False)
    with pd.ExcelWriter(RES / "experiments.xlsx") as w:
        S.to_excel(w, sheet_name="汇总")
        for f, nm in ((RES / "design_check.csv", "设计日校核"), (RES / "stress_all.csv", "压力测试"), (RES / "ops_summary.csv", "同容量运行"), (RES / "bench_all.csv", "求解效率"),
                      (RES / "calibration.csv", "模糊集校准")):
            if f.exists(): pd.read_csv(f).to_excel(w, sheet_name=nm, index=False)
    cols = ["状态", "间隙", "N", "PV", "EB", "FC", "MC", "规划能源相关费用", "复核能源相关费用", "实时SLA", "延期SLA", "失约日比例", "修复成本"]
    print(S[[c for c in cols if c in S]].to_string())


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "plan": job(sys.argv[2])
    elif w == "ops": ops()
    elif w == "bench": bench(*sys.argv[2:4])
    elif w == "collect": collect()
    elif w == "design": design_check()
    elif w == "list": print("\n".join(PLANS))
