"""温区模式：提取、去重、常值模式、初始聚类（执行方案 2.3、2.4(a)）。
模式 U 为 (J, T) 的 0-1 矩阵，每个时段恰有一个温区为 1。"""
import pickle, numpy as np


def mode_of(R):
    """由某典型日的运行解 R 取温区模式。"""
    return np.rint(np.asarray(R["u"])).astype(int)


def key(U):
    return np.asarray(U, np.int8).tobytes()


def const_modes(J, T):
    """J 个全天同一温区的常值模式（定温基线；保证任一天至少有一个可行模式）。"""
    return [np.eye(J, dtype=int)[:, [j] * T] for j in range(J)]


def is_valid(U):
    U = np.asarray(U)
    return set(np.unique(U)) <= {0, 1} and np.all(U.sum(0) == 1)


def dedup(Us):
    """按出现顺序去重，返回 (去重后的模式列表, 原序号 → 去重后序号)。"""
    out, idx, seen = [], [], {}
    for U in Us:
        k = key(U)
        if k not in seen: seen[k] = len(out); out.append(np.asarray(U, int))
        idx.append(seen[k])
    return out, idx


def from_results(paths):
    """从结果 pkl 中提取全部日模式，返回 [(模式, 来源说明)]，只取含温区变量（热耦合场景）的结果。"""
    out = []
    for p in paths:
        r = pickle.load(open(p, "rb"))
        for s, R in enumerate(r["R"]):
            if "u" in R: out.append((mode_of(R), f"{p.split('/')[-1][:-4]}#{s}"))
    return out


def mean_zone(U):
    """平均温区序号（对称性破除、排序用）。"""
    U = np.asarray(U); return float((np.arange(U.shape[0])[:, None] * U).sum() / U.shape[1])


def kmedoids_modes(Us, K, it=50):
    """按汉明距离做 K-中心点聚类，返回 (中心模式序号, 各模式所属簇)。用作初始模式库。"""
    Us = np.asarray(Us); n = len(Us)
    Dm = np.array([[np.abs(a - b).sum() / 2 for b in Us] for a in Us])
    med = [int(np.argmin(Dm.sum(1)))]                       # 最远优先初始化
    while len(med) < min(K, n): med.append(int(np.argmax(Dm[:, med].min(1))))
    for _ in range(it):
        lab = np.argmin(Dm[:, med], 1)
        new = [int(np.flatnonzero(lab == k)[np.argmin(Dm[np.ix_(lab == k, lab == k)].sum(1))]) for k in range(len(med))]
        if new == med: break
        med = new
    return med, np.argmin(Dm[:, med], 1)
