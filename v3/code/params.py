"""
表A–表D 全部参数，以及由参数预先计算的温区常数（式(4)(11)(13)(16)(18)）。
单位：功率 kW，电量 kWh，冷/热量 kW_th、kWh_th，温度 ℃，费用 元。
调参只改本文件，每次改动同步记录到 CHANGES。
"""
from copy import deepcopy
import numpy as np

# 调参记录（验收不达标时按执行方案第二节最后一列，在给定区间内调整）
CHANGES = [
    "采暖季 10-15–04-15 → 11-01–03-31",
    "实时切除惩罚 c_R 30 → 150 元/(GPU·h)",
    "自然冷却安装上限 12 → 20 MW_th",
    "式(12) N^max 取 6000",
]

BASE = dict(
    T=24, DT=1.0, DAYS=365,
    # ---------------- 表A GPU 频率档位 ----------------
    TDP=0.7, P_BASE=0.45, P_LK0=0.14, T_REF_LK=40.0,
    GEARS=[  # (频率比, 吞吐比 v, 动态功率 p_dyn, 芯片发热 h)
        (1.0, 1.00, 0.560, 0.700),
        (0.9, 0.94, 0.448, 0.588),
        (0.8, 0.87, 0.350, 0.490),
        (0.7, 0.78, 0.266, 0.406),
        (0.6, 0.68, 0.196, 0.336),
    ],
    K_R=[0, 1, 2], K_B=[0, 1, 2, 3, 4],            # 实时任务可用档位 1–3，可延期任务 1–5
    # ---------------- 表B 热管理 ----------------
    T_CHIP_MAX=85.0, R_TH=60.0 / 1000 * 1000,      # 60 K/kW（= 0.06 K/W）
    ZONES=[(30, 35), (35, 40), (40, 45), (45, 50), (50, 55)],
    ALPHA_LK=0.02, BETA=0.75, C_W=1150.0,
    DT_APP=5.0, DT_DES=15.0, A_FC=0.04,
    COP0=6.0, C1=0.12, C2=0.10, T_AMB_REF=25.0, T_W_REF=35.0, COP_MIN=3.0, COP_MAX=9.0,
    T_SUP=70.0, ETA_C=0.45, COP_HP_MAX=7.0,
    TES_LOSS=0.005, TES_EFF=0.97, TES_TAU=4.0, TES_SOC0=0.5,
    DELTA_E=0.03, A_CIRC=0.02, A_AIR=0.25, P0=300.0,
    # 无热耦合场景（S1、S2）的常规设计口径（方案A）：按 45 ℃ 温水液冷设计（ASHRAE W45 级），
    # 冷却逐时瞬时平衡，自然冷却系数按 45–50 ℃ 温区由逐时气温计算，冷机 COP 取设计日值
    # （设计气温取全年 0.4% 最高小时气温），漏电取参考值，不施加门控
    T_W_DES=45.0, AMB_DES_Q=0.996,
    # ---------------- 电池（表C 外的常规取值） ----------------
    B_ETA_C=0.95, B_ETA_D=0.95, B_LOSS=0.0002, B_SOC_MIN=0.1, B_SOC_MAX=0.9, B_SOC0=0.5, C_B_MAX=0.5,
    # ---------------- 表C 设备经济参数 (单价, 寿命, 年运维率, 安装上限) ----------------
    DISC=0.08,
    COST=dict(
        N=(250000.0, 5, 0.02, 8000),
        PV=(3000.0, 25, 0.01, 15000.0),
        WT=(4500.0, 20, 0.02, 10000.0),
        EB=(600.0, 12, 0.015, 80000.0),
        PB=(300.0, 12, 0.015, 20000.0),
        FC=(800.0, 20, 0.02, 20000.0),      # 上限由 12 MW 提至 20 MW（用户修订）
        MC=(1200.0, 20, 0.02, 10000.0),
        TES=(150.0, 25, 0.01, 60000.0),
        HP=(2000.0, 20, 0.02, 6000.0),
    ),
    # ---------------- 表D 价格、碳交易与惩罚 ----------------
    PRICE_PEAK=1.05, PRICE_FLAT=0.62, PRICE_VALLEY=0.28,
    PEAK_H=list(range(8, 11)) + list(range(17, 22)),     # 8–11 时、17–22 时
    VALLEY_H=[23] + list(range(0, 7)),                    # 23–7 时
    C_SELL=0.28, C_HEAT=0.20, P_GRID_MAX=12000.0,
    E_GRID=0.57e-3, E_HEAT=0.22e-3,                       # t/kWh、t/kWh_th
    QUOTA=0.6e-3, CO2_PRICE=80.0, CO2_LEN=20.0, CO2_GROW=0.25, CO2_V=5,   # t/(GPU·h)、元/t、t
    C_R=150.0, C_O=3.0, C_CUR=0.1, C_DEG=0.02,            # 实时切除惩罚由 30 提至 150 元/(GPU·h)（用户修订）
    D_DEADLINE=6,
    B_PERIODIC=True,      # 可延期积压按周期条件 B_1 = B_{T+1}（式(7')(8')）；False 为原式(7)(8) 日末清零
    # ---------------- 模糊集 ----------------
    K=12, M=365, ALPHA1=0.5, ALPHAINF=0.99,
    # ---------------- 负荷与热负荷 ----------------
    N_CAP=8000, PEAK_RATIO=0.85, RB_RATIO=(0.6, 0.4),
    H_PEAK=3000.0, H_SUMMER=300.0,
    # ---------------- 求解 ----------------
    N_BIGM=6000.0,        # 式(12) 中的 N^max（用户修订：6000）
    GATE_NEXT=True,       # 式(12)第二行（时段末温区安全）；诊断时可关闭
    RELAX_COMPL=True,     # 蓄冷、储能、购售电互斥 0-1 变量松弛（同时充放、同时购售无经济收益），求解后校验
    MIP_GAP=1e-4, TIME_LIMIT=1800.0,
)


def params(**over):
    P = deepcopy(BASE)
    for k, v in over.items():
        if k.startswith("CAP_"):              # 安装上限覆盖，如 CAP_PV=40000
            c = list(P["COST"][k[4:]]); c[3] = v; P["COST"][k[4:]] = tuple(c); continue
        if k not in P: raise KeyError(k)
        P[k] = v
    return P


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def annual_cost(P):
    """式(28)：单位容量年化费用 (元/单位/年)。"""
    return {k: crf(P["DISC"], v[1]) * v[0] + v[2] * v[0] for k, v in P["COST"].items()}


def xmax(P):
    return {k: v[3] for k, v in P["COST"].items()}


def price(P):
    c = np.full(P["T"], P["PRICE_FLAT"])
    c[P["PEAK_H"]] = P["PRICE_PEAK"]; c[P["VALLEY_H"]] = P["PRICE_VALLEY"]
    return c


def theta(P, M=None, K=None, a1=None, ainf=None):
    """式(31)：综合范数模糊集半径。"""
    M = M or P["M"]; K = K or P["K"]; a1 = P["ALPHA1"] if a1 is None else a1; ainf = P["ALPHAINF"] if ainf is None else ainf
    return K / (2 * M) * np.log(2 * K / (1 - a1)), 1 / (2 * M) * np.log(2 * K / (1 - ainf))


def zone_consts(P):
    """与温区有关、与气象无关的常数：漏电式(4)、安全温区式(11)、热泵 COP 式(18)。"""
    lo = np.array([z[0] for z in P["ZONES"]], float); hi = np.array([z[1] for z in P["ZONES"]], float)
    mid = (lo + hi) / 2
    plk = P["P_LK0"] * (1 + P["ALPHA_LK"] * (hi - P["T_REF_LK"]))
    h = np.array([g[3] for g in P["GEARS"]])
    safe = (hi[None, :] + P["R_TH"] * h[:, None]) <= P["T_CHIP_MAX"] + 1e-9   # [k, j]
    cop_hp = np.minimum(P["COP_HP_MAX"], P["ETA_C"] * (P["T_SUP"] + 273.15) / (P["T_SUP"] - mid))
    return dict(lo=lo, hi=hi, mid=mid, plk=plk, safe=safe, cop_hp=cop_hp)


def weather_consts(P, Tamb):
    """与气象有关的温区常数：自然冷却可用系数式(13)、冷机 COP 式(16)。Tamb: (..., T)，返回 (..., J, T)。"""
    Z = zone_consts(P); lo = Z["lo"][:, None]
    Ta = np.asarray(Tamb)[..., None, :]
    phi = np.clip((lo - P["DT_APP"] - Ta) / P["DT_DES"], 0.0, 1.0)
    cop = np.clip(P["COP0"] - P["C1"] * (Ta - P["T_AMB_REF"]) + P["C2"] * (lo - P["T_W_REF"]), P["COP_MIN"], P["COP_MAX"])
    return phi, cop


if __name__ == "__main__":
    P = params(); Z = zone_consts(P)
    print("漏电 p_lk,j:", Z["plk"].round(3))
    print("热泵 COP_j:", Z["cop_hp"].round(2))
    print("安全温区 [档位 × 温区]:\n", Z["safe"].astype(int))
    print("θ1, θ∞ =", np.round(theta(P), 4))
    print("年化单价:", {k: round(v) for k, v in annual_cost(P).items()})
    print("满载 IT 功率 (MW):", P["N_CAP"] * (P["P_BASE"] + P["GEARS"][0][2] + P["P_LK0"]) / 1000)
