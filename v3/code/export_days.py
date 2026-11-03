"""
图7–图10 数据：固定 S5 规划容量，求解 6 月 15 日（夏季）与一个冬季典型日的运行问题，导出逐时数据。
  <site>_<日期>_档位.csv      : 24 h × 5 档工作 GPU 数（另列实时/可延期分项）
  <site>_<日期>_逐时.csv      : 气温、液冷温度、温区、IT 功率、冷源出力（自然冷却/机械制冷/热泵取热）、蓄冷充放与库存、
                                热泵供热与热需求、光伏可用/实用/弃光、购电、储能充放与 SOC、电价、实时到达/处理/切除、
                                可延期到达/处理/积压/超期
用法：python export_days.py <site> <S5结果pkl> [冬季日期]      缺省冬季日为采暖季中 p0 最大的典型日
"""
import os, sys, pickle
os.environ.setdefault("HIGHS_THREADS", "1")
import numpy as np, pandas as pd
import params as PR, build_op as BO
from milp import MILP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solve(P, D, x, i, scen="S5"):
    fl = BO.flags_of(scen); M = MILP(); X, _ = BO.add_capacity(M, P, fl, fix=x)
    V = BO.add_day(M, P, X, BO.day_of(D, i), fl, w=1.0, Tamb_des=BO.tamb_design(D, P))
    sol, st, obj, g = M.solve(gap=1e-4, tl=1200)
    return BO.extract(sol, V), obj, st


def export(site, r, date, P, D, out):
    i = list(D["dates"]).index(date); R, obj, st = solve(P, D, r["x"], i); T = P["T"]
    gear = pd.DataFrame(index=pd.Index(range(T), name="时"))
    for a, k in enumerate(P["K_R"]): gear[f"实时_档{k + 1}"] = R["mR"][a].sum(0)
    for a, k in enumerate(P["K_B"]): gear[f"可延期_档{k + 1}"] = R["mB"][a].sum(0)
    for k in range(5): gear[f"档{k + 1}"] = gear.filter(like=f"档{k + 1}").sum(1)
    c = R["_c"]; g = D["gPV"][i] * r["x"]["PV"]
    h = pd.DataFrame({
        "时": range(T), "气温": D["Tamb"][i], "液冷温度": R["Tw"][:T], "温区": R["u"].argmax(0) + 1, "P_IT": R["PIT"],
        "自然冷却": R["qFC"].sum(0), "机械制冷": R["qMC"].sum(0), "热泵取热": R["qHP"].sum(0), "蓄冷充冷": R["qch"], "蓄冷放冷": R["qdis"],
        "蓄冷库存": R["S"][:T], "热泵供热": R["HHP"], "热需求": D["H"][i], "热泵用电": R["PHP"], "冷机用电": R["PMC"], "干冷器用电": R["PFC"],
        "光伏可用": g, "光伏实用": R["PV"], "弃光": R["cur"], "购电": R["buy"], "储能充电": R["Pch"], "储能放电": R["Pdis"], "储能SOC": R["SB"][:T],
        "电价": c["price"], "实时到达": D["AR"][i], "实时处理": R["UR"], "实时切除": R["LR"],
        "可延期到达": D["AB"][i], "可延期处理": R["UB"], "可延期积压": R["B"][:T], "超期量": R["O"]})
    gear.to_csv(f"{out}/{site}_{date}_档位.csv", encoding="utf-8-sig")
    h.to_csv(f"{out}/{site}_{date}_逐时.csv", index=False, encoding="utf-8-sig")
    print(f"{date}: 状态 {st}, 日运行成本 {obj / 1e4:.2f} 万元, 实时切除 {R['LR'].sum():.0f}, 超期 {R['O'].sum():.0f}, "
          f"T_w {R['Tw'][:T].min():.1f}–{R['Tw'][:T].max():.1f} ℃", flush=True)


if __name__ == "__main__":
    site, f = sys.argv[1], sys.argv[2]
    r = pickle.load(open(f, "rb")); P = PR.params(**r.get("params_over", {})); D = dict(np.load(f"{ROOT}/data/{site}_data.npz"))
    if len(sys.argv) > 3: wd = sys.argv[3]
    else:
        med = np.array(r["days"]); heat = D["heating"][med]
        wd = str(D["dates"][med[heat][np.argmax(np.asarray(r["p0"])[heat])]])
    out = f"{ROOT}/results/fig_data"; os.makedirs(out, exist_ok=True)
    for d in ("2023-06-15", wd): export(site, r, d, P, D, out)
