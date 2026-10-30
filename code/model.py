"""
模型构建与求解 —— 与论文式(1)–(18)逐条对应
  第一阶段 x = (N, K_PV, P_B, E_B, K_FC, K_MC)                          式(14)(15)
  第二阶段 每个 (d,s) 情景日的运行变量 y_ds 与约束                       式(1)–(13),(16)
  概率层: SP(经验分布) / DRO(式17 模糊集, 对偶重构) / RO(最坏情景)          式(17)(18)
求解器: HiGHS (highspy, 开源 MILP); 可替换为 Gurobi/CPLEX (只需改 solve())。
"""
import numpy as np, scipy.sparse as sp, highspy
from scipy.optimize import linprog
import config as C

INF = highspy.kHighsInf
VARIANTS = {   # 名称: (GPU运行调节, 液冷热状态耦合, 概率模型)
    "Baseline": (False, False, "SP"),   # 传统解耦规划: 仅标称模式 + 固定低温供水 + 经验分布期望
    "M1-调频":  (True,  False, "SP"),
    "M2-热耦合": (False, True,  "SP"),
    "M3-SP":    (True,  True,  "SP"),
    "M4-DRO":   (True,  True,  "DRO"),  # 本文方法
    "M5-RO":    (True,  True,  "RO"),
    "M6-DRO+极端": (True, True, "DRO"),  # 本文方法 + 极端训练日扩充有限支持集
}


class MILP:
    """极简稀疏建模器: var() 申请变量块, con() 以向量化方式添加约束块。"""
    def __init__(s):
        s.lb, s.ub, s.c, s.it = [], [], [], []
        s.r, s.cidx, s.v, s.rl, s.ru, s.nr = [], [], [], [], [], 0
        s.n = 0
        s.extra = []          # 追加目标系数 (索引, 系数)

    def add_cost(s, idx, coef):
        s.extra.append((np.ravel(idx), np.broadcast_to(coef, np.shape(np.ravel(idx))).astype(float)))

    def var(s, shape, lb=0.0, ub=INF, integer=False, cost=0.0):
        shape = (shape,) if np.isscalar(shape) else tuple(shape)
        k = int(np.prod(shape)); idx = np.arange(s.n, s.n + k).reshape(shape); s.n += k
        s.lb.append(np.broadcast_to(lb, shape).ravel().astype(float))
        s.ub.append(np.broadcast_to(ub, shape).ravel().astype(float))
        s.c.append(np.broadcast_to(cost, shape).ravel().astype(float))
        s.it.append(np.full(k, int(integer)))
        return idx

    def con(s, terms, lo=-INF, hi=INF):
        """terms: [(变量索引数组, 系数)], 各项按 numpy 广播到同一形状, 每个元素一行。"""
        shape = np.broadcast_shapes(*[np.shape(i) for i, _ in terms], *[np.shape(a) for _, a in terms],
                                    np.shape(lo), np.shape(hi))
        m = int(np.prod(shape)) if shape else 1
        rows = s.nr + np.arange(m).reshape(shape) if shape else np.array(s.nr)
        for idx, a in terms:
            I = np.broadcast_to(idx, shape).ravel(); A = np.broadcast_to(a, shape).ravel().astype(float)
            R = np.broadcast_to(rows, shape).ravel(); keep = A != 0
            s.r.append(R[keep]); s.cidx.append(I[keep]); s.v.append(A[keep])
        s.rl.append(np.broadcast_to(lo, shape).ravel().astype(float))
        s.ru.append(np.broadcast_to(hi, shape).ravel().astype(float)); s.nr += m

    def row(s, terms, lo=-INF, hi=INF):
        """单行约束: Σ_terms Σ_元素 系数·变量 ∈ [lo, hi]。"""
        I = np.concatenate([np.ravel(i) for i, _ in terms])
        A = np.concatenate([np.broadcast_to(a, np.shape(np.ravel(i))).astype(float) for i, a in terms])
        keep = A != 0
        s.r.append(np.full(keep.sum(), s.nr)); s.cidx.append(I[keep]); s.v.append(A[keep])
        s.rl.append(np.array([lo], float)); s.ru.append(np.array([hi], float)); s.nr += 1

    def solve(s, gap=C.MIP_GAP, tl=C.TIME_LIMIT, verbose=False):
        A = sp.csc_matrix((np.concatenate(s.v), (np.concatenate(s.r), np.concatenate(s.cidx))), shape=(s.nr, s.n))
        lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = s.n, s.nr
        cost = np.concatenate(s.c)
        for i, a in s.extra: np.add.at(cost, i, a)
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, np.concatenate(s.lb), np.concatenate(s.ub)
        lp.row_lower_, lp.row_upper_ = np.concatenate(s.rl), np.concatenate(s.ru)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        it = np.concatenate(s.it)
        if it.any(): lp.integrality_ = [highspy.HighsVarType(int(i)) for i in it]
        h = highspy.Highs(); h.setOptionValue("output_flag", verbose)
        h.setOptionValue("mip_rel_gap", gap); h.setOptionValue("time_limit", tl)
        h.setOptionValue("mip_abs_gap", C.MIP_ABS_GAP)
        h.setOptionValue("threads", int(__import__("os").environ.get("HIGHS_THREADS", 4))); h.passModel(lp); h.run()
        st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
        sol = np.array(h.getSolution().col_value) if info.primal_solution_status == 2 else None
        gapv = abs(info.objective_function_value - info.mip_dual_bound) if it.any() else 0.0   # 绝对间隙 元
        return sol, st, info.objective_function_value, gapv


def thermal_tables(Tamb, zones, P=None):
    """式(8)(9)系数: 自然冷却可用容量系数下界 φ_jt, 机械制冷单位电耗 a^MC_jt (均取温区下界, 不高估冷却能力)。
      φ_jt = clip((T̲_j − ΔT_CDU − T_amb,t) / ITD_des, 0, 1)
      COP_jt = min(COP_max, η_c·T_e/(T_c − T_e)),  T_e = T̲_j − ΔT_CDU − ΔT_evap,  T_c = T_amb + ΔT_cond,
      η_c 由 ASHRAE 90.1 AHRI 工况 (出水 6.7 ℃, 环境 35 ℃, COP 2.96) 标定。"""
    P = P or C.params()
    Tlo = np.array([z[0] for z in zones])[:, None]
    phi = np.clip((Tlo - P["DT_CDU"] - Tamb[None, :]) / P["ITD_DES"], 0.0, 1.0)
    K = 273.15
    eta = P["COP_AHRI"] / ((6.7 - P["DT_EVAP"] + K) / ((35.0 + P["DT_COND"]) - (6.7 - P["DT_EVAP"])))
    Te = Tlo - P["DT_CDU"] - P["DT_EVAP"] + K; Tc = Tamb[None, :] + P["DT_COND"] + K
    cop = np.where(Tc - Te > 1e-6, eta * Te / np.maximum(Tc - Te, 1e-6), P["COP_MAX"])
    cop = np.clip(cop, 1.0, P["COP_MAX"])
    return phi, 1.0 / cop, np.full_like(phi, P["A_FC"])


def spec_of(variant):
    """模型规格 (GPU调节 a, 冷却模型, 概率层)。冷却模型: 'dyn' 液冷热状态+温区+热安全 (b=1);
    'fixed' 固定低温供液温区的运行策略; 'static' 修订版 b=0 静态冷却 (无热惯性, 季节—时段冻结性能系数)。"""
    if isinstance(variant, tuple): return variant
    reg, coup, prob = VARIANTS[variant]
    return reg, ("dyn" if coup else "fixed"), prob


def build(blocks, variant, x_fix=None, slack=False, price=None, n_d=None, amb=None, cfg=None, _ro_cuts=None, x_lb=None):
    """blocks: [{'d','s','AR','AB','Tamb','gpv'[, 'Tstat']}...]; amb: 每类 d 的模糊集参数 (DRO/SP 概率)。
    variant: VARIANTS 中的名称或 (a, 冷却模型, 概率层) 元组; x_lb: 容量下界 (修复诊断用)。
    返回 (模型, 变量索引字典)。cfg 用于敏感性分析覆盖 config 中的参数。"""
    P = C.params(cfg)
    reg, cool, prob = spec_of(variant); coup = cool == "dyn"
    M = MILP(); V = {}
    if price is None: raise ValueError("必须给出分时电价 price (式16)")
    # ---------------- 第一阶段容量 式(14)(15) ----------------
    crf = lambda L: P["DISC"] * (1 + P["DISC"]) ** L / ((1 + P["DISC"]) ** L - 1)
    a = {k: v[0] * crf(v[1]) + v[2] for k, v in P["COST"].items()}       # 单位年化费用 a_i (元/单位/年)
    X = P["XMAX"]
    if x_fix is None:
        lb = x_lb or {}
        V["N"] = M.var(1, round(lb.get("N", 0)), X["N"], integer=True, cost=a["N"])
        for k in ["PV", "PB", "EB", "FC", "MC"]: V[k] = M.var(1, min(lb.get(k, 0.0), X[k]), X[k], cost=a[k])
        M.con([(V["PB"], 1.0), (V["EB"], -P["C_B_MAX"])], hi=0.0)            # 式(12) P_B ≤ c_B^max E_B
    else:
        for k in ["N", "PV", "PB", "EB", "FC", "MC"]: V[k] = M.var(1, x_fix[k], x_fix[k])
    V["a"] = a
    modes = {"R": P["K_R"] if reg else ["k1_标称"], "B": P["K_B"] if reg else ["k1_标称"]}
    J = list(range(len(P["ZONES"]))) if coup else [P["FIXED_ZONE"]]
    zones = [P["ZONES"][j] for j in J]
    Tw0 = P["TW0"] if coup else P["TW0_FIXED"]
    psl, Tt, T1, dt = P["P_SLEEP"], P["T"], P["T"] + 1, P["DT"]
    kappa_aux = 1 + P["DELTA_E"] + P["A_CIRC"] * P["BETA_LIQ"] + P["A_AIR"] * (1 - P["BETA_LIQ"])  # 式(13)
    Nmax = X["N"]
    V["blocks"] = []
    for b in blocks:
        y = {}
        # ---------- 式(1) GPU 资源、处理能力与 IT 功率 ----------
        ml = [(r, k) for r in "RB" for k in modes[r]]
        vv = np.array([P["MODES"][k][0] for _, k in ml]); pk = np.array([P["MODES"][k][1] + P["P_HOST"] for _, k in ml])
        # 有效上界(不切除最优解): 因 p_k > p^sl, 最优解中不会出现超过任务需求的运行 GPU
        cumB = np.cumsum(b["AB"])
        if P["TIGHT_BIGM"]:
            mub = np.array([np.minimum(Nmax, (b["AR"] if r == "R" else cumB) / (dt * P["MODES"][k][0])) for r, k in ml])
        else:                                                                   # 论文原式: 大M = N^max
            mub = np.full((len(ml), Tt), float(Nmax))
        y["m"] = M.var((len(ml), Tt), 0.0, mub); y["ml"] = ml
        y["U"] = {r: M.var(Tt) for r in "RB"}
        y["PIT"] = M.var(Tt)
        M.con([(y["m"][i], 1.0) for i in range(len(ml))] + [(V["N"], -1.0)], hi=0.0)
        for r in "RB":
            ii = [i for i, (rr, _) in enumerate(ml) if rr == r]
            M.con([(y["U"][r], 1.0)] + [(y["m"][i], -dt * vv[i]) for i in ii], hi=0.0)
        M.con([(y["PIT"], 1.0), (V["N"], -psl)] + [(y["m"][i], -(pk[i] - psl)) for i in range(len(ml))], 0.0, 0.0)
        # ---------- 式(2)(3) 实时任务与可延期任务积压 ----------
        y["Bq"] = M.var(T1); M.con([(y["Bq"][1:], 1.0), (y["Bq"][:-1], -1.0), (y["U"]["B"], 1.0)], b["AB"], b["AB"])
        cumA = np.concatenate([[0.0], np.cumsum(b["AB"])])
        D = P["D_DEADLINE"]; tt = np.arange(D + 1, Tt + 1)                     # t > D
        if slack:
            y["sR"] = M.var(Tt); y["sD"] = M.var(len(tt) + 1)
            M.con([(y["U"]["R"], 1.0), (y["sR"], 1.0)], b["AR"], b["AR"])
            for n, t in enumerate(tt):
                M.row([(y["U"]["B"][:t], 1.0), (y["sD"][n], 1.0)], lo=cumA[t - D])
            M.row([(y["U"]["B"], 1.0), (y["sD"][-1], 1.0)], lo=cumA[-1])        # 日末清零
            M.con([(y["Bq"][0], 1.0)], 0.0, 0.0)
        else:
            M.con([(y["U"]["R"], 1.0)], b["AR"], b["AR"])
            for t in tt: M.row([(y["U"]["B"][:t], 1.0)], lo=cumA[t - D])
            M.con([(y["Bq"][[0, Tt]], 1.0)], 0.0, 0.0)                          # B_ds1 = B_ds,T+1 = 0
        if cool == "static":
            # ---------- 修订版 b=0: 即时热量移除 Q^FC+Q^MC=β P^IT, Q^FC ≤ φ̄_dt K^FC, Q^MC ≤ K^MC ----------
            Td = P["TW_STATIC"]; Ts = np.asarray(b.get("Tstat", b["Tamb"]), float)
            phi, aMC, aFC = thermal_tables(Ts, [(Td, Td)], P)
            y["qFC"] = M.var((1, Tt)); y["qMC"] = M.var((1, Tt))
            M.con([(y["qFC"][0], 1.0), (y["qMC"][0], 1.0), (y["PIT"], -P["BETA_LIQ"])], 0.0, 0.0)
            M.con([(y["qFC"][0], 1.0), (V["FC"], -phi[0])], hi=0.0)
            M.con([(y["qMC"][0], 1.0), (V["MC"], -1.0)], hi=0.0)
            J, zones = [-1], [(Td, Td)]
        else:
            phi = None
        if cool != "static":
          # ---------- 式(4)(5) 液冷环路热动态与温区选择 ----------
          y["Tw"] = M.var(T1, zones[0][0], zones[-1][1]); y["u"] = M.var((len(J), T1), 0, 1, integer=coup)
          y["qFC"] = M.var((len(J), Tt)); y["qMC"] = M.var((len(J), Tt))
          M.con([(y["Tw"][[0, Tt]], 1.0)], Tw0, Tw0)
          M.con([(y["Tw"][1:], P["C_W"]), (y["Tw"][:-1], -P["C_W"]), (y["PIT"], -dt * P["BETA_LIQ"])]
                + [(y["qFC"][j], dt) for j in range(len(J))] + [(y["qMC"][j], dt) for j in range(len(J))], 0.0, 0.0)
          M.con([(y["u"][j], 1.0) for j in range(len(J))], 1.0, 1.0)
          M.con([(y["Tw"], 1.0)] + [(y["u"][j], -zones[j][0]) for j in range(len(J))], lo=0.0)
          M.con([(y["Tw"], 1.0)] + [(y["u"][j], -zones[j][1]) for j in range(len(J))], hi=0.0)
          # ---------- 式(6)(7) GPU 模式热安全门控 (两端点) ----------
          for i, (r, k) in enumerate(ml):
              h_k = P["MODES"][k][1]
              safe = [jj for jj, z in enumerate(zones) if z[1] + P["R_TH"] * h_k <= P["T_CHIP_MAX"] + 1e-9]
              if len(safe) == len(J): continue                                     # 全部温区安全, 约束冗余
              if not safe: M.con([(y["m"][i], 1.0)], hi=0.0); continue
              for off in (0, 1):
                  M.con([(y["m"][i], 1.0)] + [(y["u"][jj, off:Tt + off], -mub[i]) for jj in safe], hi=0.0)
          # ---------- 式(8)(9) 自然冷却与机械制冷 ----------
          phi, aMC, aFC = thermal_tables(b["Tamb"], zones, P)
          # 有效上界: 由热平衡式(7), Σq ≤ β P^IT,max + C_w (T̄_w − T̲_w)/Δt, 用于收紧温区门控的大M (不切除任何可行解)
          Hq = P["BETA_LIQ"] * Nmax * float(pk.max()) + P["C_W"] * (zones[-1][1] - zones[0][0]) / dt
          for j in range(len(J)):
              M.con([(y["qFC"][j], 1.0), (V["FC"], -phi[j])], hi=0.0)
              M.con([(y["qFC"][j], 1.0), (y["u"][j, :Tt], -np.minimum(phi[j] * X["FC"], Hq))], hi=0.0)
              M.con([(y["qMC"][j], 1.0), (V["MC"], -1.0)], hi=0.0)
              M.con([(y["qMC"][j], 1.0), (y["u"][j, :Tt], -min(X["MC"], Hq))], hi=0.0)
        # ---------- 式(10)–(12) 光伏、电网、储能 ----------
        y["PPV"] = M.var(Tt); y["Pg"] = M.var(Tt, 0, P["P_GRID_MAX"])
        M.con([(y["PPV"], 1.0), (V["PV"], -b["gpv"])], hi=0.0)
        y["S"] = M.var(T1); y["Pch"] = M.var(Tt); y["Pdis"] = M.var(Tt); y["bb"] = M.var(Tt, 0, 1, integer=not P["RELAX_B"])
        M.con([(y["S"][1:], 1.0), (y["S"][:-1], -(1 - P["DELTA_B"])), (y["Pch"], -dt * P["ETA_C"]),
               (y["Pdis"], dt / P["ETA_D"])], 0.0, 0.0)
        M.con([(y["S"], 1.0), (V["EB"], -P["SIG_MAX"])], hi=0.0); M.con([(y["S"], 1.0), (V["EB"], -P["SIG_MIN"])], lo=0.0)
        M.con([(y["S"][[0, Tt]], 1.0), (V["EB"], -P["SIG0"])], 0.0, 0.0)
        for q in ("Pch", "Pdis"): M.con([(y[q], 1.0), (V["PB"], -1.0)], hi=0.0)
        M.con([(y["Pch"], 1.0), (y["bb"], -X["PB"])], hi=0.0)
        M.con([(y["Pdis"], 1.0), (y["bb"], X["PB"])], hi=X["PB"])
        # ---------- 式(13) 电功率平衡 ----------
        M.con([(y["Pg"], 1.0), (y["PPV"], 1.0), (y["Pdis"], 1.0), (y["Pch"], -1.0), (y["PIT"], -kappa_aux)]
              + [(y["qFC"][j], -aFC[j]) for j in range(len(J))] + [(y["qMC"][j], -aMC[j]) for j in range(len(J))],
              P["P0"], P["P0"])
        if P.get("EPS_MC", 0) > 0: M.add_cost(y["qMC"], P["EPS_MC"] * dt)   # 退化平局打破 (不计入 C^op)
        # ---------- 式(16) 情景日运行费用 ----------
        y["Cop"] = M.var(1, -INF, INF)
        M.row([(y["Cop"], 1.0), (y["Pg"], -dt * price), (y["Pch"], -dt * P["C_VAR_B"]), (y["Pdis"], -dt * P["C_VAR_B"])], 0.0, 0.0)
        y.update(b=b, phi=phi, aMC=aMC, aFC=aFC, J=J, zones=zones, kaux=kappa_aux, pk=pk, vv=vv, P0=P["P0"])
        V["blocks"].append(y)
    # ---------------- 式(17)(18) 概率层 ----------------
    if slack:  # 样本外评估: 未满足工作量以大罚系数"优先满足", 单独报告, 不计入费用
        for y in V["blocks"]:
            M.add_cost(y["sR"], P["PEN_R"]); M.add_cost(y["sD"], P["PEN_B"])     # 字典序: 实时 ≫ 延期 ≫ 运行费用
    nd = n_d
    if prob == "SP" or amb is None:
        for y in V["blocks"]: M.add_cost(y["Cop"], nd[y["b"]["d"]] * y["b"]["p"])
    elif prob == "RO" and _ro_cuts is not None:   # 割平面主问题: η_d ≥ Σ_s p^(i)_ds C^op_ds
        V["eta"] = M.var(len(nd), -INF, INF, cost=nd)
        for d, plist in _ro_cuts.items():
            ys = [y for y in V["blocks"] if y["b"]["d"] == d]
            for p in plist:
                M.row([(V["eta"][d], 1.0)] + [(y["Cop"], -p[y["b"]["s"]]) for y in ys], lo=0.0)
    elif prob == "RO":
        V["eta"] = M.var(len(nd), -INF, INF, cost=nd)
        for y in V["blocks"]: M.con([(V["eta"][y["b"]["d"]], 1.0), (y["Cop"], -1.0)], lo=0.0)
    else:  # DRO: max_{p∈P_d} Σ p Q 的 LP 对偶 (强对偶, P_d 非空有界)
        V["dual"] = []
        for d in range(len(nd)):
            A = amb[d]; nf, nl = len(A["mu"]), len(A["v_p"])
            lam = M.var(1, -INF, INF, cost=nd[d])
            al = M.var(nf, cost=nd[d] * (A["mu"] + A["delta"])); be = M.var(nf, cost=-nd[d] * (A["mu"] - A["delta"]))
            gp = M.var(nl, cost=nd[d] * A["v_p"]); gn = M.var(nl, cost=nd[d] * A["v_n"])
            V["dual"].append((lam, al, be, gp, gn))
            for y in V["blocks"]:
                if y["b"]["d"] != d: continue
                s = y["b"]["s"]
                M.row([(lam, 1.0), (al, A["F"][s]), (be, -A["F"][s]), (gp, A["rho_p"][s]), (gn, A["rho_n"][s]),
                       (y["Cop"], -1.0)], lo=0.0)
    return M, V


def worst_case_p(Q, A):
    """给定各情景运行费用 Q_s, 求式(17) 模糊集上的最坏概率 p* (线性规划)。"""
    S = len(Q); F = A["F"]
    Aub = np.vstack([F.T, -F.T, A["rho_p"].T, A["rho_n"].T])
    bub = np.concatenate([A["mu"] + A["delta"], -(A["mu"] - A["delta"]), A["v_p"], A["v_n"]])
    r = linprog(-np.asarray(Q), A_ub=Aub, b_ub=bub, A_eq=np.ones((1, S)), b_eq=[1.0], bounds=(0, 1), method="highs")
    return r.x


def extract(sol, V):
    g = lambda idx: sol[np.asarray(idx)]
    x = {k: float(g(V[k])[0]) for k in ["N", "PV", "PB", "EB", "FC", "MC"]}
    out = []
    for y in V["blocks"]:
        m = g(y["m"]); qFC, qMC = g(y["qFC"]), g(y["qMC"])
        if "u" in y: zid = np.array(y["J"])[g(y["u"]).argmax(0)]; Tw = g(y["Tw"])
        else: zid = np.full(len(y["b"]["AR"]) + 1, -1); Tw = np.full(len(y["b"]["AR"]) + 1, y["zones"][0][0])
        r = dict(d=y["b"]["d"], s=y["b"]["s"], Cop=float(g(y["Cop"])[0]),
                 PIT=g(y["PIT"]), Tw=Tw, zone=zid, qFC=qFC.sum(0), qMC=qMC.sum(0),
                 PFC=(y["aFC"] * qFC).sum(0), PMC=(y["aMC"] * qMC).sum(0),
                 PPV=g(y["PPV"]), PVav=x["PV"] * y["b"]["gpv"], Pg=g(y["Pg"]), S=g(y["S"]),
                 Pch=g(y["Pch"]), Pdis=g(y["Pdis"]), UR=g(y["U"]["R"]), UB=g(y["U"]["B"]), Bq=g(y["Bq"]),
                 AR=y["b"]["AR"], AB=y["b"]["AB"], Tamb=y["b"]["Tamb"], kaux=y["kaux"],
                 modes={f"{r_}|{k}": m[i] for i, (r_, k) in enumerate(y["ml"])})
        r["Paux"] = (y["kaux"] - 1) * r["PIT"]
        r["Ptot"] = y["kaux"] * r["PIT"] + r["PFC"] + r["PMC"] + y["P0"]
        for k in ("sR", "sD"):
            if k in y: r[k] = g(y[k])
        out.append(r)
    return x, out


def solve_cutting_plane(blocks, price, n_d, amb, variant="M4-DRO", cfg=None, tol=1.0, max_it=30, tl=C.TIME_LIMIT):
    """论文第5节的迭代解法 (用于交叉验证对偶重构):
    主问题  min C_plan + Σ_d n_d η_d,  s.t. η_d ≥ Σ_s p^(i)_ds C^op_ds  (已生成的最坏分布 p^(i)),  式(1)–(16)
    子问题  p* = argmax_{p∈P_d} Σ_s p_ds C^op_ds  (LP)
    收敛    Σ_d n_d (Σ_s p*_ds C^op_ds − η_d) ≤ tol, 否则把 p* 作为新割平面加入主问题。"""
    S = {d: sum(b["d"] == d for b in blocks) for d in range(len(n_d))}
    hist, cuts = [], {d: [worst_case_p(np.zeros(S[d]), amb[d])] for d in range(len(n_d))}   # 初始割: P_d 中任一可行分布
    for it in range(max_it):
        M, V = build(blocks, "M5-RO", price=price, n_d=n_d, amb=amb, cfg=cfg, _ro_cuts=cuts)   # 物理部分与 M4 相同
        sol, st, obj, g = M.solve(tl=tl)
        x, out = extract(sol, V)
        eta = sol[V["eta"]]
        viol, ub_add = 0.0, 0.0
        for d in range(len(n_d)):
            Q = np.array([o["Cop"] for o in out if o["d"] == d])
            p = worst_case_p(Q, amb[d]); w = float(p @ Q)
            viol += n_d[d] * (w - eta[d]); ub_add += n_d[d] * w
            cuts[d].append(p)
        ub = obj - float(n_d @ eta) + ub_add                       # 当前 x 的真实最坏期望目标 (上界)
        hist.append(dict(迭代=it + 1, 主问题下界=obj, 当前解上界=ub, 违背量=viol, 绝对间隙=g))
        if viol <= tol: break
    return x, hist
