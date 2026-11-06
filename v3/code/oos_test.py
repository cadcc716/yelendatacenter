"""
表G 样本外测试：固定 SP、DRO、RO 三种容量，对全年 365 个历史日逐日求解运行问题，
统计期望能源侧成本、最高日成本、实时任务切除概率与超期量。也用于表F（S2 容量代入 S3 完整模型）。
用法：python oos_test.py <site> <方案标签 ...>     结果存 results/<site>_oos_<tag>.csv
"""
import os, sys, pickle
os.environ.setdefault("HIGHS_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import params as PR, build_op as BO
from milp import MILP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(args):
    site, scen, x, i = args[:4]; over = args[4] if len(args) > 4 else {}
    P = PR.params(**over); D = dict(np.load(f"{ROOT}/data/{site}_data.npz")); fl = BO.flags_of(scen)
    M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, w=1.0, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-3, tl=300)
    R = BO.extract(sol, V)
    act = R["mR"].sum((0, 1)) + R["mB"].sum((0, 1)); gl = act >= x["N"] - 1        # GPU 已满载的时段
    return dict(day=i, date=str(D["dates"][i]), Q=obj, LR=R["LR"].sum(), AR=D["AR"][i].sum(), LR_h=int((R["LR"] > 1).sum()),
                LR_gpu=float(R["LR"][gl].sum()), LR_cool=float(R["LR"][~gl].sum()),
                O=R["O"].sum(), buy=R["buy"].sum(), CO2=float((P["E_GRID"] * R["buy"] - P["E_HEAT"] * R["HHP"]).sum()),
                Tmax=float(D["Tamb"][i].max()), status=st)


if __name__ == "__main__":
    site = sys.argv[1]; tags = sys.argv[2:]
    src = os.environ.get("SRC", site)                  # 容量来源（极端日试验：SRC=zhongwei_ext，365 日数据仍用 site）
    rdir = os.environ.get("RDIR", f"{ROOT}/results")
    D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    for tag in tags:
        r = pickle.load(open(f"{rdir}/{src}_{tag}.pkl", "rb"))
        scen = "S5" if tag.endswith(("S5", "S6", "S7")) else tag.split("_")[-1]
        with Pool(int(os.environ.get("PAR", 4))) as p:
            res = p.map(one, [(site, scen, r["x"], i) for i in range(365)], chunksize=4)
        pd.DataFrame(res).to_csv(f"{rdir}/{src}_oos_{tag}.csv", index=False)
        print(tag, "完成", flush=True)
