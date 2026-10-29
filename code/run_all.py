"""
实验主程序: 基准与不同模型场景 -> 物理一致性校验 -> 样本外评估 -> 敏感性分析 -> 结果导出
用法:
  python run_all.py main          # 6 个模型场景 + 校验 + 样本外评估
  python run_all.py sens          # 敏感性分析 (基于 M4-DRO)
  python run_all.py all
输出: results/*.csv, results/results.xlsx, results/*.pkl
"""
import sys, time, pickle, copy
import numpy as np, pandas as pd
import config as C, model as Mo, data_prep as DP

RES = C.RES; RES.mkdir(exist_ok=True)


def load(kappa=1.0):
    f = C.PROC / "scenarios.pkl"
    if kappa != 1.0 or not f.exists():
        D = DP.build(kappa=kappa)
        if kappa == 1.0: pickle.dump(D, open(f, "wb"))
        return D
    return pickle.load(open(f, "rb"))


def blocks_of(D):
    s = D["scen"]
    return [dict(d=d, s=j, p=s["p"][d, j], AR=s["AR"][d, j], AB=s["AB"][d, j], Tamb=s["Tamb"][d, j], gpv=s["gpv"][d, j])
            for d in range(len(D["n_d"])) for j in range(s["p"].shape[1])]


def amb_of(D):
    s = D["scen"]
    return [{k: s[k][d] for k in ["F", "mu", "delta", "rho_p", "rho_n", "v_p", "v_n"]} for d in range(len(D["n_d"]))]


def plan(D, variant, cfg=None, gap=C.MIP_GAP, tl=C.TIME_LIMIT):
    t0 = time.time()
    M, V = Mo.build(blocks_of(D), variant, price=D["price"], n_d=D["n_d"], amb=amb_of(D), cfg=cfg)
    sol, st, obj, g = M.solve(gap=gap, tl=tl)
    if sol is None: raise RuntimeError(f"{variant}: {st}")
    x, out = Mo.extract(sol, V)
    if Mo.VARIANTS[variant][2] != "SP":
        # DRO/RO 中非最坏情景的运行量不影响目标(退化解); 按式(18) Q_ds(x)=min C_op 固定 x 逐情景重调度
        for i, b in enumerate(blocks_of(D)):
            Mb, Vb = Mo.build([b], variant, x_fix=x, price=D["price"], n_d=np.ones(4), cfg=cfg)
            sb, stb, _, _ = Mb.solve(gap=1e-6, tl=120)
            out[i] = Mo.extract(sb, Vb)[1][0]
    return dict(variant=variant, x=x, out=out, obj=obj, gap=g, status=st, time=time.time() - t0, a=V["a"],
                nvar=M.n, nrow=M.nr)


def summarize(R, D, cfg=None):
    """费用分解与运行指标 (经验概率 p̂ 加权年化; DRO 另给最坏分布 p*)。"""
    P = dict(vars(C)); P.update(cfg or {})
    x, a, out, nd = R["x"], R["a"], R["out"], D["n_d"]
    s = D["scen"]; amb = amb_of(D)
    capex = {k: a[k] * x[k] for k in ["N", "PV", "PB", "EB", "FC", "MC"]}
    Q = np.array([[o["Cop"] for o in out if o["d"] == d] for d in range(len(nd))])
    pw = np.array([Mo.worst_case_p(Q[d], amb[d]) for d in range(len(nd))])
    E_sp = float((nd * (s["p"] * Q).sum(1)).sum()); E_wc = float((nd * (pw * Q).sum(1)).sum())
    w = {(o["d"], o["s"]): nd[o["d"]] * s["p"][o["d"], o["s"]] for o in out}
    ann = lambda key: sum(w[(o["d"], o["s"])] * o[key].sum() for o in out) * P["DT"]
    E_IT, E_tot, E_g = ann("PIT"), ann("Ptot"), ann("Pg")
    E_pv, E_pva = ann("PPV"), ann("PVav")
    E_FC, E_MC = ann("PFC"), ann("PMC")
    q_FC, q_MC = ann("qFC"), ann("qMC")
    modes = {}
    for o in out:
        for k, v in o["modes"].items(): modes[k] = modes.get(k, 0) + w[(o["d"], o["s"])] * v.sum()
    tot_m = sum(modes.values())
    row = dict(模型=R["variant"], 状态=R["status"], 间隙=R["gap"], 求解时间s=R["time"], 变量数=R["nvar"], 约束数=R["nrow"],
               GPU台数=x["N"], 光伏kWp=x["PV"], 储能kW=x["PB"], 储能kWh=x["EB"], 自然冷却kWth=x["FC"], 机械制冷kWth=x["MC"],
               **{f"年化_{k}": v for k, v in capex.items()},
               年化投资=sum(capex.values()), 期望电费_经验=E_sp, 期望电费_最坏=E_wc,
               总费用_经验=sum(capex.values()) + E_sp, 总费用_最坏=sum(capex.values()) + E_wc, 目标值=R["obj"],
               能源相关费用_经验=sum(capex[k] for k in ["PV", "PB", "EB", "FC", "MC"]) + E_sp,
               IT电量MWh=E_IT / 1e3, 总电量MWh=E_tot / 1e3, 购电量MWh=E_g / 1e3, PUE=E_tot / E_IT,
               光伏利用率=E_pv / max(E_pva, 1e-9), 自然冷却散热占比=q_FC / max(q_FC + q_MC, 1e-9),
               冷却电量MWh=(E_FC + E_MC) / 1e3, 机械制冷电量MWh=E_MC / 1e3,
               **{f"模式占比_{k}": v / tot_m for k, v in modes.items()})
    return row, pw


def checks(R, D, cfg=None):
    """物理一致性校验: 能量守恒、热平衡、SOC、任务完成、芯片温度、容量。返回最大违背量。"""
    P = dict(vars(C)); P.update(cfg or {})
    x, rows = R["x"], []
    for o in R["out"]:
        bal = o["Pg"] + o["PPV"] + o["Pdis"] - o["kaux"] * o["PIT"] - o["PFC"] - o["PMC"] - P["P0"] - o["Pch"]
        th = P["C_W"] * np.diff(o["Tw"]) - P["DT"] * (P["BETA_LIQ"] * o["PIT"] - o["qFC"] - o["qMC"])
        soc = o["S"] / max(x["EB"], 1e-9)
        Tchip = []
        for k, m in o["modes"].items():
            h = P["MODES"][k.split("|")[1]][1]
            act = m > 1e-6
            if act.any(): Tchip.append(np.maximum(o["Tw"][:-1], o["Tw"][1:])[act].max() + P["R_TH"] * h)
        gpu = sum(o["modes"].values())
        rows.append(dict(d=o["d"], s=o["s"], 功率平衡残差kW=np.abs(bal).max(), 热平衡残差kWh=np.abs(th).max(),
                         SOC_min=soc.min() if x["EB"] > 1 else np.nan, SOC_max=soc.max() if x["EB"] > 1 else np.nan,
                         SOC首末差=abs(soc[0] - soc[-1]) if x["EB"] > 1 else 0, 实时缺额=np.abs(o["UR"] - o["AR"]).max(),
                         日末积压=o["Bq"][-1], 最高芯片温度=max(Tchip) if Tchip else np.nan,
                         GPU占用上限违背=(gpu - x["N"]).max(), 同时充放电h=int(((o["Pch"] > 1e-3) & (o["Pdis"] > 1e-3)).sum()),
                         光伏超限=(o["PPV"] - o["PVav"]).max(), 冷源超限=max((o["qFC"] - x["FC"]).max(), (o["qMC"] - x["MC"]).max()),
                         液冷温度min=o["Tw"].min(), 液冷温度max=o["Tw"].max()))
    return pd.DataFrame(rows)


def evaluate(R, D, cfg=None, n_per=None):
    """样本外评估: 固定容量 x, 在测试联合日(与训练日不重叠)上逐日求解运行问题(带缺额松弛)。"""
    te, rows = D["test"], []
    for d in range(len(D["n_d"])):
        n = len(te["AR"][d]) if n_per is None else min(n_per, len(te["AR"][d]))
        for i in range(n):
            b = dict(d=d, s=0, p=1.0, AR=te["AR"][d][i], AB=te["AB"][d][i], Tamb=te["Tamb"][d][i], gpv=te["gpv"][d][i])
            M, V = Mo.build([b], R["variant"], x_fix=R["x"], slack=True, price=D["price"], n_d=np.ones(4), cfg=cfg)
            sol, st, obj, g = M.solve(gap=0.005, tl=60)
            if sol is None: rows.append(dict(d=d, i=i, 状态=st)); continue
            _, o = Mo.extract(sol, V); o = o[0]
            rows.append(dict(d=d, i=i, 状态=st, 日运行费用=o["Cop"], 实时缺额=o["sR"].sum(), 延期缺额=o["sD"].sum(),
                             Tmax=b["Tamb"].max(), PV日=b["gpv"].sum(), B日=b["AB"].sum(),
                             风险得分=float(D["risk_A"].sum(0) @ te["F"][d][i])))
    return pd.DataFrame(rows)


def oos_summary(E, D):
    nd = D["n_d"]; out = {}
    ok = E.dropna(subset=["日运行费用"])
    per = ok.groupby("d")["日运行费用"].mean()
    out["样本外年运行费用"] = float((nd * per.reindex(range(4)).values).sum())
    # 分布偏移: 指数倾斜权重 w ∝ exp(θ·风险得分), 模拟高温/低光伏/高负荷日更频繁
    for th in (0.5, 1.0):
        tot = 0
        for d in range(4):
            g = ok[ok.d == d]; w = np.exp(th * g["风险得分"].values); w /= w.sum()
            tot += nd[d] * (w * g["日运行费用"].values).sum()
        out[f"偏移θ={th}年运行费用"] = float(tot)
    c = ok["日运行费用"].values; q = np.quantile(c, 0.9)
    out["日费用CVaR90"] = float(c[c >= q].mean())
    out["缺额日比例"] = float(((ok["实时缺额"] + ok["延期缺额"]) > 1e-3).mean())
    out["实时缺额GPUh"] = float(ok["实时缺额"].sum()); out["延期缺额GPUh"] = float(ok["延期缺额"].sum())
    return out


def export_ts(R, tag):
    rows = []
    for o in R["out"]:
        for t in range(C.T):
            r = dict(模型=tag, d=o["d"], s=o["s"], t=t, 电价=None, 环境温度=o["Tamb"][t], 实时到达=o["AR"][t],
                     延期到达=o["AB"][t], 实时完成=o["UR"][t], 延期完成=o["UB"][t], 积压=o["Bq"][t], IT功率=o["PIT"][t],
                     液冷温度=o["Tw"][t], 温区=o["zone"][t], 自然冷却散热=o["qFC"][t], 机械制冷散热=o["qMC"][t],
                     自然冷却电耗=o["PFC"][t], 机械制冷电耗=o["PMC"][t], 光伏可用=o["PVav"][t], 光伏利用=o["PPV"][t],
                     购电=o["Pg"][t], 充电=o["Pch"][t], 放电=o["Pdis"][t], SOC电量=o["S"][t], 总负荷=o["Ptot"][t])
            for k, v in o["modes"].items(): r[f"m_{k}"] = v[t]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    D0 = load(); DX = pickle.load(open(C.PROC / "scenarios_ext.pkl", "rb"))
    rs, ck, ts, oos, pws = [], [], [], [], {}
    for v in Mo.VARIANTS:
        D = DX if "极端" in v else D0
        R = plan(D, v); pickle.dump(R, open(RES / f"plan_{v}.pkl", "wb"))
        row, pw = summarize(R, D); pws[v] = pw
        c = checks(R, D); c.insert(0, "模型", v); ck.append(c)
        t = export_ts(R, v); t["电价"] = np.tile(D["price"], len(R["out"])); ts.append(t)
        E = evaluate(R, D); E.insert(0, "模型", v); E.to_csv(RES / f"oos_{v}.csv", index=False)
        row.update(oos_summary(E, D)); rs.append(row)
        print(f"[{v}] {R['status']} obj={R['obj']:.4e} N={R['x']['N']:.0f} time={R['time']:.0f}s", flush=True)
    S = pd.DataFrame(rs); K = pd.concat(ck); TS = pd.concat(ts)
    base = S.iloc[0]
    for col in ["总费用_经验", "总费用_最坏", "能源相关费用_经验", "期望电费_经验", "购电量MWh", "冷却电量MWh", "样本外年运行费用",
                "偏移θ=1.0年运行费用"]:
        S[col + "_较基准%"] = (S[col] / base[col] - 1) * 100
    S.to_csv(RES / "summary.csv", index=False); K.to_csv(RES / "checks.csv", index=False); TS.to_csv(RES / "timeseries.csv", index=False)
    P = pd.DataFrame([dict(模型=v, 类别=D0["classes"][d], 情景=s, 经验概率=(DX if "极端" in v else D0)["scen"]["p"][d, s],
                           最坏概率=pws[v][d, s]) for v in pws for d in range(4) for s in range(pws[v].shape[1])])
    P.to_csv(RES / "probabilities.csv", index=False)
    with pd.ExcelWriter(RES / "results.xlsx") as w:
        S.to_excel(w, sheet_name="汇总", index=False); K.to_excel(w, sheet_name="校验", index=False); P.to_excel(w, sheet_name="概率", index=False)
    print(S.T.to_string())


def sens():
    """敏感性分析: 每次只改变一个参数, 其余保持基准; 模型为 M4-DRO (另给 M3-SP 对照)。"""
    cases = [("kappa", k, None) for k in (0.0, 0.5, 2.0, 4.0)]
    cases += [("FLOAT", f, {"FLOAT": f}) for f in (0.3, 0.8)]                    # 宁夏征求意见稿: 新能源±30%, 尖峰深谷±80%
    cases += [("R_TH", r, {"R_TH": C.R_TH * r}) for r in (0.9, 1.1)]
    cases += [("BETA_LIQ", b, {"BETA_LIQ": b}) for b in (0.6, 0.8)]                # 冷板捕获 60–80%
    cases += [("C_W", c, {"C_W": c}) for c in (10.0, 90.0)]
    rows = []
    D0 = load()
    for name, val, cfg in cases:
        D = load(kappa=val) if name == "kappa" else D0
        if name == "FLOAT":
            D = copy.deepcopy(D0); k = np.ones(24); k[C.PEAK_H] += val; k[C.VALLEY_H] -= val
            D["price"] = np.roll(C.LAMBDA_E * k + C.C_TD, -C.H0)
        for v in (["M4-DRO", "M3-SP"] if name in ("FLOAT", "R_TH") else ["M4-DRO"]):
            R = plan(D, v, cfg=cfg)
            row, _ = summarize(R, D, cfg); row.update(参数=name, 取值=val); rows.append(row)
            print(f"[sens] {name}={val} {v} obj={R['obj']:.4e}", flush=True)
            pd.DataFrame(rows).to_csv(RES / "sensitivity.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("main", "all"): main()
    if what in ("sens", "all"): sens()
