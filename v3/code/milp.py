"""极简稀疏 MILP 建模器（向量化添加变量块与约束块），求解器 HiGHS。"""
import os
import numpy as np, scipy.sparse as sp, highspy

INF = highspy.kHighsInf


class MILP:
    """极简稀疏建模器: var() 申请变量块, con() 以向量化方式添加约束块。"""
    def __init__(s):
        s.lb, s.ub, s.c, s.it = [], [], [], []
        s.r, s.cidx, s.v, s.rl, s.ru, s.nr = [], [], [], [], [], 0
        s.n = 0
        s.extra = []          # 追加目标系数 (索引, 系数)
        s.int_extra = []      # 事后指定为整数的变量索引
        s.dual_bound = None

    def set_int(s, idx):
        s.int_extra.append(np.ravel(idx))

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

    def solve(s, gap=1e-4, tl=1800.0, verbose=False, start=None, abs_gap=1.0):
        A = sp.csc_matrix((np.concatenate(s.v), (np.concatenate(s.r), np.concatenate(s.cidx))), shape=(s.nr, s.n))
        lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = s.n, s.nr
        cost = np.concatenate(s.c)
        for i, a in s.extra: np.add.at(cost, i, a)
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, np.concatenate(s.lb), np.concatenate(s.ub)
        lp.row_lower_, lp.row_upper_ = np.concatenate(s.rl), np.concatenate(s.ru)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        it = np.concatenate(s.it)
        for ix in s.int_extra: it[ix] = 1
        if it.any(): lp.integrality_ = [highspy.HighsVarType(int(i)) for i in it]
        h = highspy.Highs(); h.setOptionValue("output_flag", verbose)
        h.setOptionValue("mip_rel_gap", gap); h.setOptionValue("time_limit", tl)
        h.setOptionValue("mip_abs_gap", abs_gap)
        h.setOptionValue("threads", int(__import__("os").environ.get("HIGHS_THREADS", 4))); h.passModel(lp)
        if start is not None:   # 热启动: 同结构模型的可行解 (列顺序一致)
            hs = highspy.HighsSolution(); hs.col_value = list(np.asarray(start, float)); hs.value_valid = True; h.setSolution(hs)
        h.run()
        st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
        sol = np.array(h.getSolution().col_value) if info.primal_solution_status == 2 else None
        s.dual_bound = info.mip_dual_bound if it.any() else info.objective_function_value
        gapv = abs(info.objective_function_value - info.mip_dual_bound) if it.any() else 0.0   # 绝对间隙 元
        return sol, st, info.objective_function_value, gapv
