"""
单个典型日的运行约束 式(1)–(29)，以及容量变量 式(28)(33)。
开关 flags（执行方案 5.2 节“✕”的实现方式）：
  dvfs     False → GPU 固定第 1 档
  thermal  True  → 热耦合：温区门控 + 温区漏电 + 气象相关自然冷却与冷机 COP + 液冷热惯性 C_w
           False → 按 45 ℃ 温水液冷设计（方案A）：C_w = 0 冷却瞬时平衡，自然冷却系数按 45–50 ℃ 温区由逐时气温计算，
                    冷机 COP 取设计日值，漏电取参考值，不施加门控
  tes      False → 不配蓄冷（E_TES = 0）
  heat     False → K_HP = 0
场景可行域嵌套：S1 ⊂ S2，S3 ⊂ S4 ⊂ S5。园区禁止向电网反送电（防逆流）；不配置风电。
"""
import numpy as np
import params as PR
from milp import MILP, INF

CAPS = ["N", "PV", "PB", "EB", "FC", "MC", "TES", "HP"]
SCEN = {   # 名称: (dvfs, thermal, tes, heat)
    "S1": (False, False, False, False),
    "S2": (True, False, False, False),
    "S3": (True, True, False, False),
    "S4": (True, True, True, False),
    "S5": (True, True, True, True),
}


def flags_of(name):
    d, t, c, h = SCEN.get(name, SCEN["S5"])
    return dict(dvfs=d, thermal=t, tes=c, heat=h)


def add_capacity(M, P, flags, fix=None, integer_N=True, fixN=None):
    """容量变量与投资费用 式(28)(33)。fix: 固定全部容量（单日测试、子问题(36)、样本外测试）；
    fixN: 两步法第二步，固定 GPU 数量且不计其年化投资（常数项），目标只含能源侧年化成本。"""
    a = PR.annual_cost(P); xm = PR.xmax(P); X = {}
    ub = dict(xm)
    if not flags["tes"]: ub["TES"] = 0.0
    if not flags["heat"]: ub["HP"] = 0.0
    for k in CAPS:
        if fix is not None:
            v = float(fix.get(k, 0.0)); X[k] = M.var(1, v, v)
        elif k == "N" and fixN is not None:
            X[k] = M.var(1, float(fixN), float(fixN))
        else:
            X[k] = M.var(1, 0.0, ub[k], integer=(k == "N" and integer_N), cost=a[k])
    if fix is None:
        M.con([(X["PB"], 1.0), (X["EB"], -P["C_B_MAX"])], hi=0.0)
    return X, a


def add_day(M, P, X, day, flags, w=1.0, Tamb_des=None, int_add=None):
    """在模型 M 中加入一个典型日的运行变量与约束，返回变量索引字典 V；
    V['Q'] 为该日运行成本 C_op（式(29)，元/日），以权重 w 计入目标。"""
    T = P["T"]; G = P["GEARS"]; J = len(P["ZONES"]); Z = PR.zone_consts(P)
    KR, KB = P["K_R"], P["K_B"]
    AR, AB, Ta, gpv, Hd = (np.asarray(day[k], float) for k in ("AR", "AB", "Tamb", "gPV", "H"))
    Nmax = P["N_BIGM"]; V = {}
    # ---------- 温区相关常数 ----------
    if flags["thermal"]:
        phi, cop = PR.weather_consts(P, Ta)                       # (J, T)
        plk = Z["plk"]; safe = Z["safe"]
        zone_ok = np.ones(J, bool)
    else:
        jd = int(np.where(Z["lo"] == P["T_W_DES"])[0][0])
        phi, _ = PR.weather_consts(P, Ta)                          # 自然冷却：45–50 ℃ 温区、逐时气温
        _, cop_d = PR.weather_consts(P, np.array([Tamb_des]))
        cop = np.repeat(cop_d[:, :1], T, 1)                       # 冷机 COP：设计日值
        plk = np.full(J, P["P_LK0"]); safe = np.ones((len(G), J), bool)
        zone_ok = np.arange(J) == jd
    gear_ok = np.array([True] * len(G)) if flags["dvfs"] else (np.arange(len(G)) == 0)
    # ---------- (1)(2)(3) GPU 分配与功率 ----------
    okR = (gear_ok[KR][:, None] & safe[KR] & zone_ok[None, :]).astype(float)       # (|K_R|, J)
    okB = (gear_ok[KB][:, None] & safe[KB] & zone_ok[None, :]).astype(float)
    V["mR"] = mR = M.var((len(KR), J, T), 0.0, okR[:, :, None] * Nmax)
    V["mB"] = mB = M.var((len(KB), J, T), 0.0, okB[:, :, None] * Nmax)
    allm = np.concatenate([mR.reshape(-1, T), mB.reshape(-1, T)])
    M.con([(allm[i], 1.0) for i in range(allm.shape[0])] + [(X["N"], -1.0)], hi=0.0)
    vR = np.array([G[k][1] for k in KR]); vB = np.array([G[k][1] for k in KB])
    V["UR"] = UR = M.var(T); V["UB"] = UB = M.var(T)
    M.con([(UR, 1.0)] + [(mR[i, j], -vR[i] * P["DT"]) for i in range(len(KR)) for j in range(J)], hi=0.0)
    M.con([(UB, 1.0)] + [(mB[i, j], -vB[i] * P["DT"]) for i in range(len(KB)) for j in range(J)], hi=0.0)
    V["PIT"] = PIT = M.var(T)
    terms = [(PIT, 1.0), (X["N"], -P["P_BASE"])]
    for i, k in enumerate(KR):
        for j in range(J): terms.append((mR[i, j], -(G[k][2] + plk[j])))
    for i, k in enumerate(KB):
        for j in range(J): terms.append((mB[i, j], -(G[k][2] + plk[j])))
    M.con(terms, lo=0.0, hi=0.0)
    # ---------- (5)–(8) 算力任务 ----------
    V["LR"] = LR = M.var(T, 0.0, AR)
    M.con([(UR, 1.0), (LR, 1.0)], lo=AR, hi=AR)
    V["B"] = Bk = M.var(T + 1)
    M.con([(Bk[1:], 1.0), (Bk[:-1], -1.0), (UB, 1.0)], lo=AB, hi=AB)
    D = P["D_DEADLINE"]; cumA = np.cumsum(AB)
    V["O"] = O = M.var(T)
    if P["B_PERIODIC"]:
        # 式(8')：积压周期条件 B_1 = B_{T+1}，日末未完成任务顺延至次日凌晨（典型日按周期稳态处理）
        # 式(7')：先到先服务，结转积压 B_1 由上一周期最后到达的任务构成，到达满 D 小时即计超期
        M.con([(Bk[[T]], 1.0), (Bk[[0]], -1.0)], lo=0.0, hi=0.0)
        for t in range(T):
            if t >= D:
                M.row([(O[t], 1.0), (UB[:t + 1], 1.0), (Bk[0], -1.0)], lo=cumA[t - D])
            else:
                tail = AB[T + t - D + 1:].sum() if t - D + 1 < 0 else 0.0
                M.row([(O[t], 1.0), (UB[:t + 1], 1.0), (Bk[0], -1.0)], lo=-tail)
    else:
        M.con([(Bk[[0, T]], 1.0)], lo=0.0, hi=0.0)                         # 式(8)：日初、日末积压为零
        for t in range(D, T):
            M.row([(O[t], 1.0), (UB[:t + 1], 1.0)], lo=cumA[t - D])
    # ---------- (9)(10) 液冷热状态与温区 ----------
    V["u"] = u = M.var((J, T), 0.0, zone_ok[:, None].astype(float) * np.ones(T), integer=True)
    if flags["thermal"]:
        M.con([(u[j], 1.0) for j in range(J)], lo=1.0, hi=1.0)
    else:
        M.con([(u[jd], 1.0)], lo=1.0, hi=1.0)
    V["Tw"] = Tw = M.var(T + 1, Z["lo"].min(), Z["hi"].max())
    M.con([(Tw[:T], 1.0)] + [(u[j], -Z["lo"][j]) for j in range(J)], lo=0.0)
    M.con([(Tw[:T], 1.0)] + [(u[j], -Z["hi"][j]) for j in range(J)], hi=0.0)
    M.con([(Tw[[T]], 1.0), (Tw[[0]], -1.0)], lo=0.0, hi=0.0)                 # 周期条件 T_w,1 = T_w,T+1
    nxt = np.r_[np.arange(1, T), 0]                                          # 时段末温区（周期）
    # ---------- (12) 热安全门控 ----------
    for j in range(J):     # 第一行按温区汇总：Σ_{r,k} m_rkjt ≤ N^max·u_jt（与逐档写法整数可行域相同，松弛更紧）
        tj = [(mR[i, j], 1.0) for i, k in enumerate(KR) if okR[i, j]] + [(mB[i, j], 1.0) for i, k in enumerate(KB) if okB[i, j]]
        if tj: M.con(tj + [(u[j], -Nmax)], hi=0.0)
    for mm, KK in ((mR, KR), (mB, KB)):
        for i, k in enumerate(KK):
            if flags["thermal"] and P["GATE_NEXT"]:
                sj = [j for j in range(J) if safe[k, j]]
                M.con([(mm[i, j], 1.0) for j in range(J)] + [(u[j][nxt], -Nmax) for j in sj], hi=0.0)
    # ---------- (13)–(16) 自然冷却与机械制冷 ----------
    xm = PR.xmax(P)
    V["qFC"] = qFC = M.var((J, T)); V["qMC"] = qMC = M.var((J, T)); V["qHP"] = qHP = M.var((J, T))
    for j in range(J):
        M.con([(qFC[j], 1.0), (X["FC"], -phi[j])], hi=0.0)
        M.con([(qFC[j], 1.0), (u[j], -phi[j] * xm["FC"])], hi=0.0)
        M.con([(qMC[j], 1.0), (u[j], -xm["MC"])], hi=0.0)
        M.con([(qHP[j], 1.0), (u[j], -xm["HP"])], hi=0.0)
    M.con([(qMC[j], 1.0) for j in range(J)] + [(X["MC"], -1.0)], hi=0.0)
    V["PFC"] = PFC = M.var(T); V["PMC"] = PMC = M.var(T)
    M.con([(PFC, 1.0)] + [(qFC[j], -P["A_FC"]) for j in range(J)], lo=0.0, hi=0.0)
    M.con([(PMC, 1.0)] + [(qMC[j], -1.0 / cop[j]) for j in range(J)], lo=0.0, hi=0.0)
    # ---------- (17) 蓄冷罐 ----------
    V["S"] = S = M.var(T + 1); V["qch"] = qch = M.var(T); V["qdis"] = qdis = M.var(T)
    V["z"] = z = M.var(T, 0.0, 1.0 if flags["tes"] else 0.0, integer=flags["tes"] and not P["RELAX_COMPL"])
    Qt = xm["TES"] / P["TES_TAU"]
    M.con([(S[1:], 1.0), (S[:-1], -(1 - P["TES_LOSS"])), (qch, -P["TES_EFF"] * P["DT"]), (qdis, P["DT"] / P["TES_EFF"])], lo=0.0, hi=0.0)
    M.con([(S, 1.0), (X["TES"], -1.0)], hi=0.0)
    M.con([(qch, 1.0), (X["TES"], -1.0 / P["TES_TAU"])], hi=0.0)
    M.con([(qdis, 1.0), (X["TES"], -1.0 / P["TES_TAU"])], hi=0.0)
    M.con([(qch, 1.0), (z, -Qt)], hi=0.0)
    M.con([(qdis, 1.0), (z, Qt)], hi=Qt)
    M.con([(S[[0, T]], 1.0), (X["TES"], -P["TES_SOC0"])], lo=0.0, hi=0.0)
    # ---------- (9) 液冷环路热平衡 ----------
    Cw = P["C_W"] if flags["thermal"] else 0.0               # 液冷热惯性归入热耦合
    terms = [(PIT, P["DT"] * P["BETA"]), (qch, P["DT"]), (qdis, -P["DT"])]
    terms += [(q[j], -P["DT"]) for q in (qFC, qMC, qHP) for j in range(J)]
    if Cw > 0: terms += [(Tw[1:], -Cw), (Tw[:-1], Cw)]
    M.con(terms, lo=0.0, hi=0.0)
    # ---------- (18)(19)(25) 余热热泵 ----------
    ch = Z["cop_hp"]
    V["PHP"] = PHP = M.var(T); V["HHP"] = HHP = M.var(T, 0.0, Hd)
    M.con([(PHP, 1.0)] + [(qHP[j], -1.0 / (ch[j] - 1)) for j in range(J)], lo=0.0, hi=0.0)
    M.con([(HHP, 1.0)] + [(qHP[j], -ch[j] / (ch[j] - 1)) for j in range(J)], lo=0.0, hi=0.0)
    M.con([(HHP, 1.0), (X["HP"], -1.0)], hi=0.0)
    # ---------- (20)(21)(22)(23) 辅助电耗、风光、储能、电网 ----------
    kaux = P["DELTA_E"] + P["A_CIRC"] * P["BETA"] + P["A_AIR"] * (1 - P["BETA"])
    V["PV"] = PV = M.var(T); V["cur"] = cur = M.var(T)
    M.con([(PV, 1.0), (X["PV"], -gpv)], hi=0.0)
    M.con([(cur, 1.0), (PV, 1.0), (X["PV"], -gpv)], lo=0.0, hi=0.0)
    V["SB"] = SB = M.var(T + 1); V["Pch"] = Pch = M.var(T); V["Pdis"] = Pdis = M.var(T)
    V["b"] = b = M.var(T, 0.0, 1.0, integer=not P["RELAX_COMPL"])
    PBm = PR.xmax(P)["PB"]
    M.con([(SB[1:], 1.0), (SB[:-1], -(1 - P["B_LOSS"])), (Pch, -P["B_ETA_C"] * P["DT"]), (Pdis, P["DT"] / P["B_ETA_D"])], lo=0.0, hi=0.0)
    M.con([(SB, 1.0), (X["EB"], -P["B_SOC_MAX"])], hi=0.0)
    M.con([(SB, 1.0), (X["EB"], -P["B_SOC_MIN"])], lo=0.0)
    M.con([(Pch, 1.0), (X["PB"], -1.0)], hi=0.0); M.con([(Pdis, 1.0), (X["PB"], -1.0)], hi=0.0)
    M.con([(Pch, 1.0), (b, -PBm)], hi=0.0); M.con([(Pdis, 1.0), (b, PBm)], hi=PBm)
    M.con([(SB[[0, T]], 1.0), (X["EB"], -P["B_SOC0"])], lo=0.0, hi=0.0)
    V["buy"] = buy = M.var(T, 0.0, P["P_GRID_MAX"])            # 式(23)：0 ≤ P_buy ≤ P_grid,max，禁止反送电
    Pg = P["P_GRID_MAX"]
    # ---------- (24) 电功率平衡 ----------
    M.con([(buy, 1.0), (PV, 1.0), (Pdis, 1.0), (PIT, -(1 + kaux)), (PFC, -1.0), (PMC, -1.0), (PHP, -1.0),
           (Pch, -1.0)], lo=P["P0"], hi=P["P0"])
    # ---------- (26)(27) 阶梯碳交易 ----------
    Vn = P["CO2_V"]
    V["eps"] = eps = M.var(Vn, 0.0, np.r_[np.full(Vn - 1, P["CO2_LEN"]), INF]); V["eps_s"] = eps_s = M.var(1)
    M.row([(buy, P["E_GRID"] * P["DT"]), (HHP, -P["E_HEAT"] * P["DT"]), (UR, -P["QUOTA"]), (UB, -P["QUOTA"]),
           (eps, -1.0), (eps_s, 1.0)], lo=0.0, hi=0.0)
    # ---------- (29) 运行成本 ----------
    c = PR.price(P); lam = P["CO2_PRICE"]
    V["Q"] = Q = M.var(1, -INF, INF, cost=w)
    M.row([(Q, 1.0), (buy, -c * P["DT"]), (HHP, P["C_HEAT"] * P["DT"]),
           (cur, -P["C_CUR"] * P["DT"]), (Pch, -P["C_DEG"] * P["DT"]), (Pdis, -P["C_DEG"] * P["DT"]),
           (LR, -P["C_R"]), (O, -P["C_O"]), (eps, -lam * (1 + np.arange(Vn) * P["CO2_GROW"])), (eps_s, lam)], lo=0.0, hi=0.0)
    V["_c"] = dict(phi=phi, cop=cop, plk=plk, kaux=kaux, price=c)
    for k, ts in (int_add or {}).items():          # 校验发现同时充放/购售的时段，只对这些时段加回 0-1 变量
        M.set_int(V[k][list(ts)])
    return V


def day_of(D, i):
    return {k: D[k][i] for k in ("AR", "AB", "Tamb", "gPV", "H")}


def tamb_design(D, P):
    return float(np.quantile(D["Tamb"], P["AMB_DES_Q"]))


def extract(sol, V):
    out = {}
    for k, idx in V.items():
        if k.startswith("_"): continue
        out[k] = sol[idx]
    out["_c"] = V["_c"]
    return out
