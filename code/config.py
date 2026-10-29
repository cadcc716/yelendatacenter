"""
参数配置：考虑GPU运行调节与液冷热状态耦合的AI数据中心分布鲁棒容量规划模型
------------------------------------------------------------------------
每个参数给出: 数值、单位、来源、可信度等级、在模型中的位置(式号)。
可信度: A = 已核对一手文献/官方文件原文;  B = 来自检索摘要或二手转述, 需按原文复核;
        C = 案例设计值/由文献区间推导的标定值, 需按工程实测替换。
修改参数只需改本文件; run_all.py 会把 PARAM_TABLE 导出为论文参数表。
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES, FIG = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results", ROOT / "figures"

FX = 7.1884            # CNY/USD, 2024-12-31 人民币兑美元中间价 (CFETS, 新京报转载)  [B]
DT, T = 1.0, 24        # 调度步长 h, 时段数

# ---------------- 年度代表类别 d 与天数 n_d ----------------
SEASONS = {  # 类别: (月份, 天数 n_d), sum n_d = 365
    "春季": ([3, 4, 5], 92), "夏季": ([6, 7, 8], 92),
    "秋季": ([9, 10, 11], 91), "冬季": ([12, 1, 2], 90)}
N_SCEN = 5             # (旧版) 每类联合情景数
N_MED = 4              # 每类 k-medoids 代表日数 (修订版 5.1)
EXTREMES = ["T_max", "R_peak"]  # 每类额外保留的训练集极端日 (单独成簇, 概率 1/n_train): 最高气温日、实时负荷峰值日
DESIGN_RETURN_N = 20   # 设计极端日: ASHRAE n=20 年一遇极端最高干球温度 (关键设施冷源设计取值, Uptime Tier 标准建议) [B]
SPLIT = (0.60, 0.15)   # 每月按日期先后: 前 60% 训练, 随后 15% 验证, 其余 25% 封存测试 (修订版 5.1.2 连续区块划分)
SEED = 2024

# ---------------- GPU 运行模式 (式1, 式6) ----------------
# 工作量单位: 1 = 1台GPU以标称模式运行1 h 的处理量 (GPU·h_nom)
TDP = 0.700            # kW, H100 SXM TDP (Splitwise ISCA'24 表1)                     [A]
P_HOST = 0.467         # kW/台, 主机(CPU/内存/网卡/风扇)分摊; 由"GPU功率占服务器60%" (Patel et al. ASPLOS'24 Insight)
                       # 与 TDP 推得 0.7/0.6-0.7; 并满足 DGX H100 最大 10.2 kW/8 = 1.275 kW/台      [C]
GPU_IDLE = 0.20 * TDP  # kW, GPU 空闲约 20% TDP (ASPLOS'24 §4.1: "20%... idle power of the GPUs")    [A]
MODES = {  # 名称: (相对处理速率 v, GPU功率 kW)   ——功率为整机边界: p = p_gpu + P_HOST
    "k1_标称":     (1.00, TDP),          # 不限频
    "k2_浅锁频":   (0.95, TDP * 0.87),   # BLOOM: 5% 性能损失对应 13% 峰值功率下降 (ASPLOS'24 §4.2)   [A/C]
    "k3_深锁频":   (0.93, TDP * 0.80),   # "up to 20% power ... for up to 7% performance loss" (ASPLOS'24) [A/C]
}
K_R = ["k1_标称", "k2_浅锁频"]                      # 实时任务仅允许满足时延要求的模式 (POLCA: HP 仅轻度降频)
K_B = ["k1_标称", "k2_浅锁频", "k3_深锁频"]
P_SLEEP = GPU_IDLE + P_HOST                          # kW/台 休眠(空闲)功率, 与运行功率同一计量边界

# ---------------- 热安全 (式6) ----------------
T_CHIP_MAX = 85.0      # ℃ 允许热点温度; A100 nvidia-smi Max Operating 85℃, H100 slowdown≈87℃ 取保守值  [B]
T_INLET_SPEC = 45.0    # ℃ H100 700 W 液冷节点允许最高进水温度 (Lenovo SD665-N V3 datasheet)            [B]
R_TH = (T_CHIP_MAX - T_INLET_SPEC) / TDP        # K/kW: 标称满功率在45℃进水恰达限值 => 57.1 K/kW [C]
ZONES = [(25.0, 32.0), (32.0, 38.0), (38.0, 45.0), (45.0, 50.0), (50.0, 53.0)]  # 液冷温区 ℃ (覆盖W32~W45+) [C]
TW0 = 40.0             # ℃ 代表日初末液冷等效温度 (式5)                                                 [C]
TW0_FIXED = 30.0       # ℃ 解耦基准的固定温区中点 (W32 类常规供水)                                       [C]
FIXED_ZONE = 0         # 解耦基准固定在 ZONES[0]
TW_STATIC = 32.0       # ℃ b=0 静态冷却模型的设计供液温度 (ASHRAE W32 常规设计); 敏感性取 45 (W45 温水设计)  [C]
PEN_R, PEN_B = 1.0e5, 1.0e4   # 元/GPU·h 仅用于评价器的字典序失约诊断 (实时 ≫ 延期 ≫ 运行费用), 不进入规划目标

# ---------------- 液冷环路与冷源 (式4, 8, 9, 13) ----------------
BETA_LIQ = 0.75        # 进入液冷的热量比例; 冷板方案捕获 60–80% (Fujitsu CCLC; Asetek InRackCDU)       [B]
C_W = 30.0             # kWh/K 液冷环路等效热容(≈25.8 m3 水), 既有基础设施参数, 做敏感性                  [C]
DT_CDU = 3.0           # K CDU 换热端差 (典型 2–5 K)                                                    [C]
ITD_DES = 11.1         # K 干冷器设计进口温差/最小逼近 20°F (ACEEE 2026 高温液冷技术经济研究)            [B]
A_FC = 0.045           # kWe/kWth 干冷器风机电耗 (ACEEE 2026: 0.045 kWe/kWt)                             [B]
COP_AHRI = 10.1 / 3.412  # 风冷冷水机组 ≥150 RT 最低 EER 10.1 → COP 2.96 (ASHRAE 90.1-2019 表6.8.1-3)    [B]
COP_MAX = 8.0          # 高出水温工况 COP 上限 (模型假设, 需厂家曲线)                                     [C]
DT_EVAP, DT_COND = 3.0, 10.0  # K 蒸发/冷凝端差 (Carnot 比例标定用)                                    [C]
A_CIRC = 23.0 / 1350.0 # kWe/kWth 主循环泵: LiquidStack/Trane CDU-1MW 23 kW / 1350 kW                   [B]
A_AIR = 1.0 / COP_AHRI # kWe/kWth 空气侧剩余热量按 AHRI 工况冷水机组 COP 计                               [C]
DELTA_E = 0.08         # IT 侧配电损耗系数; 变压器+UPS+PDU 损耗占设施 8–12% (Uptime 相关报道)            [B]
P0 = 30.0              # kW 其他辅助负荷(照明、安防、办公), 案例设计值                                   [C]

# ---------------- 光伏、储能、电网 (式10–12) ----------------
PV_TILT, PV_AZ = 35.0, 180.0
PV_GAMMA = -0.0037     # 1/K PVWatts v5 标准组件温度系数 (Dobos, NREL/TP-6A20-62641, 2014)              [A]
PV_LOSS = 0.14         # PVWatts 默认系统损失 14%                                                       [A]
PV_INV_EFF = 0.96      # PVWatts 名义逆变器效率                                                         [A]
ETA_C = ETA_D = np.sqrt(0.85)  # 往返效率 85% (NREL ATB 储能技术假设)                                  [B]
DELTA_B = 0.02 / 720   # 1/h 自放电 ≈ 2%/月 (磷酸铁锂常见值)                                           [C]
SIG_MIN, SIG_MAX, SIG0 = 0.10, 0.90, 0.50  # SOC 边界与初末值                                          [C]
C_B_MAX = 0.5          # 1/h 最大倍率 (2 h 系统, 与中标价格口径一致)                                   [A]
C_VAR_B = 0.0          # 元/kWh 储能可变运维: 无可验证数据, 按模型说明不另计                             [—]

# ---------------- 容量上限 x̄ (式14) ----------------
XMAX = dict(N=2560, PV=8000.0, PB=4000.0, EB=16000.0, FC=5000.0, MC=5000.0)   # N: 320 台 8 卡服务器机位  # 案例设计值 [C]
P_GRID_MAX = 6000.0    # kW 合同/接入容量                                                                [C]

# ---------------- 经济参数 (式15) ----------------
DISC = 0.08            # 折现率; 《建设项目经济评价方法与参数》(第三版)社会折现率 8% (需核对原文)      [B]
COST = {  # 设备: (单位投资 元/单位, 寿命 年, 固定运维 元/单位/年)
    "N":  (375_000 / 8 * FX, 5, 5000 / 8 * FX),  # 每台 GPU 分摊整机 $375k/台服务器, 维护 $5000/台·年,
                                                 # 寿命 4–6 年取 5 (Microsoft Research 2026 AI-DC Lifecycle)   [A]
    "PV": (3160.0, 25, 3160.0 * 0.015),          # CPIA 2023-24 路线图: 2023 地面 3.4 元/W, 2024 预测 3.16 元/W;
                                                 # 运维 1.5%/年 为常用区间中值                                [B/C]
    "PB": (110.0, 15, 0.0),                      # 组串式 PCS ≈0.11 元/W (SMM 2024-07)                         [B]
    "EB": ((1089.02 * 2 - 110.0) / 2, 15, 0.0),  # 2 h 储能 EPC 中标均价 1089.02 元/kWh (2024-12, 财联社), 扣除PCS [B]
    "FC": (264 / 3.517 * FX, 20, 0.0),           # 干冷器 $264/ton (ACEEE 2026); 冷却塔类寿命 20 年 (ASHRAE)     [B]
    "MC": (1200 / 3.517 * FX, 23, 0.0),          # 冷水机组 $1200/ton (ACEEE 2026); 离心机组中位寿命 23 年 (ASHRAE)[B]
}

# ---------------- 电价 (式16): 宁夏工商业峰谷分时 ----------------
LAMBDA_E = 0.271 + 0.018  # 元/kWh 2024 年宁夏市场交易均价 (2025 均价 0.271, 较2024同期降1.8分; 国家能源局西北监管局) [B]
C_TD = 0.0920          # 元/kWh 两部制 1–10 kV 电量输配电价 (国网宁夏 2023 信息公开年报, 第三监管周期)  [B]
FLOAT = 0.50           # 峰/谷 在平段基础上上浮/下浮 50% (宁发改 2023 优化峰谷分时电价通知)            [A]
PEAK_H = list(range(7, 9)) + list(range(17, 23))   # 高峰 7–9, 17–23 时                                [A]
VALLEY_H = list(range(9, 17))                       # 低谷 9–17 时 (午间新能源大发)                      [A]

# ---------------- 任务与时延 (式2, 3) ----------------
R_MEAN_GPU = 500.0     # 实时任务年均需求 (GPU·h_nom/h) —— 轨迹规模化系数, 案例设计值                   [C]
B_MEAN_GPU = 200.0     # 可延期任务年均需求                                                             [C]
D_DEADLINE = 12        # h 可延期任务最大延期 (≤ Azure/OpenAI Batch 24 h 目标完成窗口)                  [B/C]
BURST_DAY0 = 9         # BurstGPT API 日志在第9天后进入稳定服务期
H0 = 6                 # 调度日窗口起点(当地时刻): 6:00–次日6:00, 避开可延期任务晚间到达高峰与日末清零冲突 [C]

# ---------------- 分布鲁棒模糊集 (式17) ----------------
FEATURES = ["T_mean", "T_max", "PV_day", "R_day", "B_day"]
RISK_DIRS = {"高温": "T_max", "低光伏": "PV_day", "高负荷": "B_day"}  # 风险方向 ℓ 与符号
N_BOOT, BOOT_Q = 2000, 0.95  # bootstrap 估计 δ_d、v±_dℓ
KAPPA = 1.0            # 模糊集缩放系数 (敏感性分析)

# ---------------- 求解器 ----------------
MIP_GAP, TIME_LIMIT = 1e-6, 1800.0
MIP_ABS_GAP = 1.0e4    # 元/年; 总费用中 GPU 投资约占 96%, 以绝对间隙控制精度 (约为能源相关费用的 0.15%)
RELAX_B = False        # 式(12) b∈{0,1} (与论文一致); True 仅用于数值试验
TIGHT_BIGM = True      # 式(7) 大M取 min(N^max, 需求界) 并给 m 加同一上界 (有效不等式, 不改变最优值, verify.py 验证);
                       # False 时严格按论文用 N^max
VALID_FRAC = 0.25      # 式(17) 训练日中用于验证集校准 δ、v± 的比例

# ---------------- 派生参数 (敏感性分析覆盖基础参数时必须联动重算) ----------------
_DERIVED = {
    "GPU_IDLE": lambda P: 0.20 * P["TDP"],
    "MODES": lambda P: {"k1_标称": (1.00, P["TDP"]), "k2_浅锁频": (0.95, P["TDP"] * 0.87),
                        "k3_深锁频": (0.93, P["TDP"] * 0.80)},
    "P_SLEEP": lambda P: P["GPU_IDLE"] + P["P_HOST"],
    "R_TH": lambda P: (P["T_CHIP_MAX"] - P["T_INLET_SPEC"]) / P["TDP"],
    "A_AIR": lambda P: 1.0 / P["COP_AHRI"],
}


def params(cfg=None):
    """返回参数字典: 先套用覆盖值 cfg, 再按依赖顺序重算未被显式覆盖的派生参数。"""
    P = {k: v for k, v in globals().items() if k.isupper()}
    cfg = dict(cfg or {}); P.update(cfg)
    for k, f in _DERIVED.items():
        if k not in cfg: P[k] = f(P)
    return P
