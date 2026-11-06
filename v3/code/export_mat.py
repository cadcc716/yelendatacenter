"""把 Python 预处理得到的数据导出为 MATLAB .mat（典型日序号转为 1 起始）。用法：python export_mat.py <输出目录>"""
import sys, os
import numpy as np
from scipy.io import savemat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def export(name, out):
    D = dict(np.load(f"{ROOT}/data/{name}_data.npz"))
    m = dict(site=name, dates=np.array([list(s) for s in D["dates"]]), AR=D["AR"], AB=D["AB"], Tamb=D["Tamb"], gPV=D["gPV"], H=D["H"],
             heating=D["heating"].astype(float), med=D["med"].astype(float).reshape(-1, 1) + 1, lab=D["lab"].astype(float).reshape(-1, 1) + 1,
             p0=D["p0"].reshape(-1, 1), theta1=float(D["theta1"]), thetainf=float(D["thetainf"]))
    if "ext_days" in D:
        m["ext_days"] = D["ext_days"].astype(float).reshape(-1, 1) + 1
        m["ext_names"] = "；".join(str(s) for s in D["ext_names"])
    savemat(f"{out}/{name}_data.mat", m, do_compression=True)
    print(name, {k: np.shape(v) for k, v in m.items() if not isinstance(v, (str, float))})

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for n in ("zhongwei", "zhongwei_ext", "guian", "guian_ext", "shaoguan", "shaoguan_ext"):
        if os.path.exists(f"{ROOT}/data/{n}_data.npz"): export(n, out)
