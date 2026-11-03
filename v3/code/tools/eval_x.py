"""固定容量，逐典型日求解运行问题并汇总服务水平（调试用）。用法：python tools/eval_x.py site scen 'json容量' [参数覆盖...]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HIGHS_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP

def one(args):
    site, scen, x, over, i = args
    P = PR.params(**over); D = dict(np.load(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/../data/{site}_data.npz"))
    fl = BO.flags_of(scen); M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, w=1.0, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-4, tl=600); R = BO.extract(sol, V)
    return i, obj, R["LR"].sum(), D["AR"][i].sum(), R["O"].sum(), D["AB"][i].sum(), R["Tw"][:24].mean()

if __name__ == "__main__":
    site, scen, x = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    over = {a.split("=")[0]: eval(a.split("=")[1]) for a in sys.argv[4:]}
    D = dict(np.load(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f"/../data/{site}_data.npz"))
    with Pool(4) as p: res = p.map(one, [(site, scen, x, over, i) for i in D["med"]])
    p0 = D["p0"]
    for (i, obj, lr, ar, o, ab, tw), q in zip(res, p0):
        print(f"{D['dates'][i]} p0={q:.3f} Q={obj/1e4:7.2f}万 实时切除 {lr:7.0f}/{ar:7.0f} ({lr/ar:5.1%}) 超期 {o:8.0f} GPU·h·h, 可延期 {ab:6.0f}, 平均Tw {tw:.1f}")
    lr = sum(r[2] * q for r, q in zip(res, p0)); ar = sum(r[3] * q for r, q in zip(res, p0))
    print(f"期望实时切除率 {lr/ar:.2%}，期望日运行费 {sum(r[1]*q for r,q in zip(res,p0))/1e4:.2f} 万元")
