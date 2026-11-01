"""
E3 补充: 风险成本型 (软约束) 规划的样本外评价
  风险调整年化费用 = 能源相关费用 (光伏/储能/冷源年化投资 + 测试日年化运行费) + c_R·年化实时失约 + c_B·年化延期超时
  - 测试日失约由统一评价器按字典序 (先尽量服务, 再最小费用) 得到, 与规划时是否用软约束无关;
  - 年化: 各季节测试日均值 × n_d;
  - 配对区块 bootstrap (修订版 5.4.1): 同一测试日的逐日差值, 以季节内连续日期区块为单位重抽样 (2000 次), 给出 95% 置信区间。
输出 results/exp/risk_table.csv, risk_boot.csv, risk_sweep.csv, risk_pw.csv
"""
import pickle
import numpy as np, pandas as pd
import config as C, exp as X

RES = C.RES / "exp"; RNG = np.random.default_rng(20261101)
HARD = {"M3": "硬约束 SP(=M3)", "U1": "硬约束 U1", "U2": "硬约束 U2", "U3": "硬约束 U3"}
SOFT = lambda c: {f"R{c}-SP": f"软约束 SP (c_R={c})", f"R{c}-U1": f"软约束 U1 (c_R={c})", f"R{c}-U2": f"软约束 U2 (c_R={c})",
                  f"R{c}-U3": f"软约束 U3 (c_R={c})"}


def load(nm):
    f = RES / f"hf_{nm}.csv"
    if not f.exists() or not (RES / f"plan_{nm}.pkl").exists(): return None
    return pd.read_csv(f), pickle.load(open(RES / f"plan_{nm}.pkl", "rb"))


def daily(E, R, c):
    """逐日风险调整费用 (元/日, 不含投资), 实时失约按 c_R、延期超时按 c_B = c_R/5 计价。"""
    return E["日运行费用"].fillna(0).values + c * E["实时失约"].fillna(0).values + c / 5 * E["延期失约"].fillna(0).values


def annual(E, v, nd):
    per = pd.Series(v).groupby(E["d"].values).mean().reindex(range(4)).values
    return float((nd * per).sum())


def summary(nm, c, nd):
    r = load(nm)
    if r is None: return None
    E, R = r; cx = X.capex(R); inv = sum(cx.values()) - cx["N"]
    op = annual(E, E["日运行费用"].fillna(0).values, nd)
    sR = annual(E, E["实时失约"].fillna(0).values, nd); sD = annual(E, E["延期失约"].fillna(0).values, nd)
    gl = annual(E, E["实时失约_GPU台数受限"].fillna(0).values, nd)
    return dict(方案=nm, 光伏=R["x"]["PV"], 自然冷却=R["x"]["FC"], 机械制冷=abs(max(R["x"]["MC"], 0)), 冷源投资=(cx["FC"] + cx["MC"]) / 1e4,
                能源投资=inv / 1e4, 年运行费用=op / 1e4, 年化实时失约=sR, 其中冷却受限=sR - gl, 年化延期超时=sD,
                失约费用=(c * sR + c / 5 * sD) / 1e4, 风险调整费用=(inv + op + c * sR + c / 5 * sD) / 1e4, 间隙=R["gap"] / 1e4)


def boot(a, b, c, nd, B=2000):
    """配对区块 bootstrap: 方案 a 相对 b 的风险调整年化费用差 (万元)。"""
    (Ea, Ra), (Eb, Rb) = load(a), load(b)
    assert (Ea[["d", "i"]].values == Eb[["d", "i"]].values).all()
    inv = lambda R: sum(X.capex(R).values()) - X.capex(R)["N"]
    dinv = inv(Ra) - inv(Rb); dd = daily(Ea, Ra, c) - daily(Eb, Rb, c)
    w = Ea["w"].values; d = Ea["d"].values
    blk = np.concatenate([[0], np.cumsum(np.diff(w) != 1)])          # 连续日期区块
    point = (dinv + annual(Ea, dd, nd)) / 1e4
    stats = []
    for _ in range(B):
        tot = dinv
        for s in range(4):
            bs = np.unique(blk[d == s]); pick = RNG.choice(bs, len(bs), replace=True)
            vals = np.concatenate([dd[blk == k] for k in pick]); tot += nd[s] * vals.mean()
        stats.append(tot / 1e4)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(比较=f"{a} − {b}", c_R=c, 差值万元=point, CI下=lo, CI上=hi, 优于概率=float(np.mean(np.array(stats) < 0)),
                日差值为负比例=float(np.mean(dd < -1e-6)), 日差值为正比例=float(np.mean(dd > 1e-6)))


if __name__ == "__main__":
    D = X.dataset(); nd = D["n_d"]
    rows = []
    for c in (30, 100, 300):
        for nm, lab in {**HARD, **SOFT(c)}.items():
            r = summary(nm, c, nd)
            if r: rows.append(dict(评价单价c_R=c, 标签=lab, **r))
    T = pd.DataFrame(rows); T.to_csv(RES / "risk_table.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(2).to_string())
    bs = []
    for c in (30, 100, 300):
        for a, b in ((f"R{c}-U3", f"R{c}-SP"), (f"R{c}-U2", f"R{c}-SP"), (f"R{c}-U3", f"R{c}-U2"), (f"R{c}-U3", f"R{c}-U1"), (f"R{c}-U3", "M3")):
            if load(a) is not None and load(b) is not None: bs.append(boot(a, b, c, nd))
    B = pd.DataFrame(bs); B.to_csv(RES / "risk_boot.csv", index=False); print(B.round(3).to_string())
    # 规划阶段: 夏季类别 (含设计日) 的经验概率与最坏分布
    pw = []
    for nm in [f"R100-{u}" for u in ("SP", "U2", "U3")] + ["M3", "U3"]:
        f = RES / f"plan_{nm}.pkl"
        if not f.exists(): continue
        R = pickle.load(open(f, "rb"))
        if nm.endswith("U1"): continue
        for d in range(4):
            bl = [b for b in R["blocks"] if b["d"] == d]; o = [x for x in R["out"] if x["d"] == d]
            p = R["pw"][d] if R["pw"] else np.array([b["p"] for b in bl])
            for b, q, pp in zip(bl, o, p):
                pw.append(dict(方案=nm, 季节=d, 情景=b["tag"], 经验概率=b["p"], 加权概率=pp, 情景费用=q["Cop"], 实时失约=q["sR"].sum() if "sR" in q else 0.0,
                               Tmax=b["Tamb"].max()))
    pd.DataFrame(pw).to_csv(RES / "risk_pw.csv", index=False)
