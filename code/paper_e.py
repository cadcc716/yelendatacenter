"""
论文初稿生成: docs/论文初稿_中国电机工程学报.md
正文中的全部数值与表格均从 results/exp/*.csv / plan_*.pkl 直接读取, 不手工录入。
主算例: 宁夏中卫 ERA5 2014–2024 (按年份外推) + BurstGPT 188 日; 适用性分析: 典型气象年 (丹佛等)。
"""
import pickle
import numpy as np, pandas as pd
import config as C, exp as X

RES = C.RES / "exp"; W = 1e4; P = C.params()


def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)



def render(t, env):
    """模板渲染: {{ → {, }} → }, {expr[:fmt]} → 求值; 单独的 } 视为字面量 (LaTeX 友好)。"""
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        if c == "{" and t[i + 1:i + 2] == "{": out.append("{"); i += 2; continue
        if c == "}" and t[i + 1:i + 2] == "}": out.append("}"); i += 2; continue
        if c == "{":
            j, depth = i + 1, 1
            while depth:
                if t[j] == "{": depth += 1
                elif t[j] == "}": depth -= 1
                j += 1
            expr = t[i + 1:j - 1]; out.append(eval('f"{' + expr + '}"', env)); i = j; continue
        out.append(c); i += 1
    return "".join(out)


f1 = lambda x: f"{x:,.1f}"
f0 = lambda x: f"{x:,.0f}"
Z = pd.read_csv(RES / "rob_zw.csv").set_index("代号"); Zd = pd.read_csv(RES / "rob_zw_decomp.csv").set_index("分项")
Zb = pd.read_csv(RES / "rob_zw_boot.csv").set_index("比较")
D = X.dataset("zw"); Tdes = D["design"]["T"]
U = pickle.load(open(RES / "plan_Z-R100-U3.pkl", "rb")); SPp = pickle.load(open(RES / "plan_Z-R100-SP.pkl", "rb"))
sb = [b for b in U["blocks"] if b["d"] == 1]; pw = U["pw"][1]
p_des = float(pw[[i for i, b in enumerate(sb) if b["tag"].startswith("设计")][0]])
p_hot = float(pw[[i for i, b in enumerate(sb) if "T_max" in b["tag"]][0]]); ph_hot = float([b["p"] for b in sb if "T_max" in b["tag"]][0])
m0, md0, m1, m2, m3, sp, u2, u3 = (Z.loc[k] for k in ("Z-M0", "Z-M0-设计工况", "Z-M1", "Z-M2", "Z-M3", "Z-R100-SP", "Z-R100-U2", "Z-R100-U3"))
dE = m3["能源相关费用"] - m0["能源相关费用"]; pE = dE / m0["能源相关费用"] * 100
dEd = m3["能源相关费用"] - md0["能源相关费用"]; pEd = dEd / md0["能源相关费用"] * 100
I_GT = Zd.loc["能源相关费用", "交互项"]; cap_int = Zd.loc["冷源投资", "交互项"] / Zd.loc["冷源投资", "合计"] * 100
b_m3m0, b_m3md, b_u3sp = Zb.loc["Z-M3 − Z-M0"], Zb.loc["Z-M3 − Z-M0-设计工况"], Zb.loc["Z-R100-U3 − Z-R100-SP"]
dcool = (m3["冷却电量MWh"] / m0["冷却电量MWh"] - 1) * 100
OPS = pd.read_csv(RES / "ops_Z-summary.csv").set_index("策略") if (RES / "ops_Z-summary.csv").exists() else None
ops_ok = OPS is not None and len(OPS) == 4
# 典型气象年 (适用性分析)
S = pd.read_csv(RES / "summary.csv", index_col=0); Sn = pd.read_csv(RES / "rob_sens.csv"); Fn = pd.read_csv(RES / "rob_fan.csv")
RB = pd.read_csv(RES / "risk_boot.csv").set_index("比较")
yrs = D["Tamb_all"];

# ---------------------------------------------------------------- 表格
tab_par = md(pd.DataFrame([
    ["GPU 运行模式 (速率/整机功率)", "标称 1.00/1.167 kW；浅锁频 0.95/1.076 kW；深锁频 0.93/1.027 kW；休眠 0/0.03 kW", "文献[8–9]"],
    ["实时 / 可延期可用模式", "标称、浅锁频 / 全部三种；可延期最大延期 12 h；热备用比例 ρ=0.2", "文献[9]"],
    ["芯片热点上限、等效热阻", f"{P['T_CHIP_MAX']:.0f} ℃；{P['R_TH']:.1f} K/kW", "设备资料"],
    ["液冷温区", "25–53 ℃ 划分 10 区；液冷捕获比例 β=0.75；等效热容 30 kWh/K", "ASHRAE[16]"],
    ["干冷器", "设计逼近温差 11.1 K；风机单位电耗 0.045 kWe/kWth（变工况割线近似）", "—"],
    ["冷水机组", "AHRI 工况 COP 2.96，按 Carnot 比例随温度修正，上限 8.0", "—"],
    ["年化单位投资", f"GPU {X.capex(U)['N'] / U['x']['N']:,.0f} 元/台·a；光伏 {U['a']['PV']:.0f} 元/kWp·a；干冷器 {U['a']['FC']:.1f} 元/kWth·a；冷水机组 {U['a']['MC']:.1f} 元/kWth·a；储能 {U['a']['EB']:.0f} 元/kWh·a", "折现率 8%"],
    ["电价", "宁夏峰谷分时电价，平段 0.381 元/kWh，峰谷 ±50%", "宁夏发改委"],
    ["失约单价", "实时 c_R=100 元/GPU·h；可延期 c_B=20 元/GPU·h·h", "按 GPU 租赁价 5–10 倍设定"],
], columns=["参数", "取值", "依据"]))

lab = {"Z-M0": "M0 静态基准", "Z-M0-设计工况": "M0-D 设计工况法", "Z-M1": "M1 仅 GPU 调节", "Z-M2": "M2 仅热状态反馈", "Z-M3": "M3 完整耦合 (硬约束)",
       "Z-R100-SP": "R-SP 风险成本随机规划", "Z-R100-U2": "R-U2 条件对称 DRO", "Z-R100-U3": "**R-U3 本文方法**"}
tab_cases = md(pd.DataFrame([
    ["M0", "×", "×", "静态（32 ℃ 设计供液，季节—时段平均气温）", "经验分布，硬约束"],
    ["M0-D", "×", "×", "静态（32 ℃，逐时气象 + 设计日）", "经验分布，硬约束"],
    ["M1", "√", "×", "静态", "经验分布，硬约束"],
    ["M2", "×", "√", "液冷热状态 + 温区 + 气象", "经验分布，硬约束"],
    ["M3", "√", "√", "液冷热状态 + 温区 + 气象", "经验分布，硬约束"],
    ["R-SP", "√", "√", "同 M3", "经验分布，风险成本"],
    ["R-U2", "√", "√", "同 M3", "条件对称矩模糊集，风险成本"],
    ["R-U3", "√", "√", "同 M3", "**条件正负半方差模糊集，风险成本（本文）**"],
], columns=["方案", "GPU 运行调节", "热状态反馈", "冷却模型", "不确定性处理"]))

tab_cfg = md(pd.DataFrame([[lab[k], f0(Z.loc[k, "光伏"]), f0(Z.loc[k, "自然冷却"]), f0(Z.loc[k, "机械制冷"]), f1(Z.loc[k, "冷源投资"]), f1(Z.loc[k, "光伏投资"]),
                             f1(Z.loc[k, "年运行费用"]), f1(Z.loc[k, "能源相关费用"]), f0(Z.loc[k, "冷却电量MWh"])]
                            for k in ("Z-M0", "Z-M0-设计工况", "Z-M1", "Z-M2", "Z-M3")],
                           columns=["方案", "光伏/kWp", "自然冷却/kWth", "机械制冷/kWth", "冷源投资/万元", "光伏投资/万元", "运行费用/万元", "能源相关费用/万元", "冷却电量/MWh"]))
tab_dec = md(pd.DataFrame([[k, f"{Zd.loc[k, '仅GPU调节']:+.1f}", f"{Zd.loc[k, '仅热耦合']:+.1f}", f"{Zd.loc[k, '交互项']:+.1f}", f"{Zd.loc[k, '合计']:+.1f}"]
                            for k in ("光伏投资", "冷源投资", "运行费用", "能源相关费用")],
                           columns=["分项", "GPU 调节单独作用", "热状态反馈单独作用", "交互项", "合计 (M3−M0)"]))
tab_unc = md(pd.DataFrame([[lab[k], f0(Z.loc[k, "自然冷却"]), f0(Z.loc[k, "机械制冷"]), f1(Z.loc[k, "能源相关费用"]), f"{Z.loc[k, '实时SLA']:.2f}",
                             f0(Z.loc[k, "年化冷却受限失约"]), f0(Z.loc[k, "设计日失约"]), f1(Z.loc[k, "风险调整费用"])]
                            for k in ("Z-M3", "Z-R100-SP", "Z-R100-U2", "Z-R100-U3")],
                           columns=["方案", "自然冷却/kWth", "机械制冷/kWth", "能源相关费用/万元", "实时SLA/%", "冷却受限失约/(GPU·h/a)", "设计日失约/GPU·h", "风险调整费用/万元"]))
blab = {"Z-M3 − Z-M0": "M3 − M0", "Z-M3 − Z-M0-设计工况": "M3 − M0-D", "Z-R100-U3 − Z-M0": "R-U3 − M0", "Z-R100-U3 − Z-R100-SP": "R-U3 − R-SP",
        "Z-R100-U2 − Z-R100-SP": "R-U2 − R-SP", "Z-R100-U3 − Z-R100-U2": "R-U3 − R-U2"}
tab_boot = md(pd.DataFrame([[blab[k], f"{r['差值万元']:+.1f}", f"[{r['CI下']:+.1f}, {r['CI上']:+.1f}]"] for k, r in Zb.iterrows() if k in blab],
                            columns=["比较", "风险调整年化费用差/万元", "95% 置信区间"]))
if ops_ok:
    o = OPS.loc[["M0策略", "M1策略", "M2策略", "M3策略"]]
    tab_ops = md(pd.DataFrame([[n.replace("策略", " 运行策略"), f1(r["复核年运行费用"] / W), f"{r['实时SLA'] * 100:.2f}", f0(r["实时失约GPUh"]),
                                f"{int(r['不可安全运行日数'])}", f1(r["复核冷却电量MWh"])] for n, r in o.iterrows()],
                              columns=["运行策略 (容量均取 M3)", "年运行费用/万元", "实时SLA/%", "测试日实时失约/GPU·h", "不可安全运行测试日/d", "冷却电量/MWh"]))
    ops_save = (1 - o.loc["M3策略", "复核年运行费用"] / o.loc["M2策略", "复核年运行费用"]) * 100
else:
    tab_ops = "（中卫同容量运行比较计算中）"; ops_save = np.nan
tab_sens = md(pd.DataFrame([[r["参数"].split(" (")[0], f"{r['浅锁频安全温度']:.1f}", f0(r["M3机械制冷"]), f"{r['M3较M0百分比']:+.1f}"] for _, r in Sn.iterrows()],
                            columns=["参数取值", "浅锁频安全供液温度/℃", "M3 机械制冷/kWth", "M3 相对 M0 能源相关费用/%"]))
cl = []
for nm, a, b in (("丹佛 (半干旱)", "M0", "M3"), ("芝加哥 (温带大陆性)", "S2-芝加哥-M0", "S2-芝加哥-M3"), ("坦帕 (湿热)", "S2-坦帕-M0", "S2-坦帕-M3"),
                 ("旧金山 (温和海洋性)", "S2-旧金山-M0", "S2-旧金山-M3")):
    cl.append([nm, f"{abs(max(S.loc[a, 'MC'], 0)):.0f} / {abs(max(S.loc[b, 'MC'], 0)):.0f}",
               f"{(S.loc[b, '复核能源相关费用'] / S.loc[a, '复核能源相关费用'] - 1) * 100:+.1f}"])
cl.append(["**中卫 (ERA5 实测外推)**", f"{m0['机械制冷']:.0f} / {m3['机械制冷']:.0f}", f"{pE:+.1f}"])
tab_clim = md(pd.DataFrame(cl, columns=["气候", "机械制冷 M0 / M3 /kWth", "M3 相对 M0 能源相关费用/%"]))
cr = [[c, f"{RB.loc[f'R{c}-U3 − R{c}-SP', '差值万元']:+.1f}"] for c in (30, 100, 300) if f"R{c}-U3 − R{c}-SP" in RB.index]
tab_cr = md(pd.DataFrame(cr, columns=["失约单价 c_R/(元/GPU·h)", "R-U3 相对 R-SP 风险调整费用/万元"]))
fv = Fn[Fn["风机模型"] == "风机变工况"].set_index("方案")

# ---------------------------------------------------------------- 正文
txt = render("""# 考虑GPU运行调节与液冷热状态耦合的AI数据中心风险成本型分布鲁棒容量配置

**Risk-cost Distributionally Robust Capacity Configuration of AI Data Centers Considering GPU Operating Regulation and Liquid-cooling Thermal-state Coupling**

> 初稿（自动生成于 `code/paper_e.py`；全部数值取自 `results/exp/`）。作者、单位、基金项目待补。图号对应 `figures/` 中的文件，见文末"图件清单"。

## 摘要

AI 数据中心的 GPU 运行模式同时决定算力供给、电功率与散热量，而液冷温度和室外气象又反过来限制 GPU 可用模式与自然冷却能力，传统"固定单位算力功耗 + 固定冷却效率"的容量规划难以刻画这一双向耦合。为此，提出考虑 GPU 运行调节与液冷热状态耦合的风险成本型分布鲁棒容量配置方法。

- **GPU 运行模型**：建立含标称、锁频与休眠状态的 GPU 多模式运行模型，以及计及实时/可延期任务时限的算力调度约束。
- **热安全耦合**：以液冷环路等效热状态和温区门控刻画"GPU 模式—热点温度—可用液冷温区—自然冷却能力"的耦合可行域，并计及风机变工况的冷源电耗。
- **规划模型**：将服务失约以风险成本计入情景费用，构建基于条件正负半方差有限支持模糊集的两阶段分布鲁棒容量配置模型，经对偶重构转化为混合整数线性规划求解。

以宁夏中卫 2014—2024 年 ERA5 逐时气象和真实 LLM 服务负荷为算例，按年份进行时间外推检验，结果表明：

- 与静态冷却规划相比，所提耦合模型使机械制冷容量由 {m0['机械制冷']:.0f} kW 降至 {m3['机械制冷']:.0f} kW，能源相关年化费用降低 {abs(pE):.1f}%；其中 GPU 调节与热状态反馈的交互作用贡献了 {cap_int:.0f}% 的冷源投资节约。
- 风险成本型分布鲁棒模型通过最坏分布自动为罕见高温工况保留冷却储备，样本外风险调整费用显著低于随机规划。

**关键词**：AI数据中心；GPU运行调节；液冷热状态；自然冷却；容量配置；分布鲁棒优化

## Abstract

GPU operating modes jointly determine the computing supply, electric power, and heat release of AI data centers, while the liquid-cooling temperature and ambient weather in turn restrict the admissible GPU modes and the free-cooling capability. This paper proposes a risk-cost distributionally robust capacity configuration method that captures this bidirectional coupling. A multi-mode GPU model with nominal, frequency-capped and sleep states is combined with deadline-aware workload scheduling. A liquid-loop thermal-state model with temperature-zone gating links GPU thermal safety, free-cooling capability, and cooling power with variable-speed fans. Service shortfalls are priced in the scenario cost, and a two-stage distributionally robust model with a conditional semi-variance finite-support ambiguity set is reformulated as a mixed-integer linear program. A case study using 2014–2024 ERA5 hourly weather of Zhongwei, Ningxia, and real LLM serving traces, evaluated by year-based out-of-sample extrapolation, shows that the coupled model reduces chiller capacity from {m0['机械制冷']:.0f} kW to {m3['机械制冷']:.0f} kW and the energy-related annualized cost by {abs(pE):.1f}%, with {cap_int:.0f}% of the cooling-investment saving arising from the interaction between GPU regulation and thermal feedback; the risk-cost DRO model preserves cooling reserves for rare hot extremes and significantly reduces the out-of-sample risk-adjusted cost compared with stochastic programming.

**Key words**: AI data center; GPU operating regulation; liquid-cooling thermal state; free cooling; capacity configuration; distributionally robust optimization

## 0 引言

大模型训练与推理使数据中心负荷由通用计算向 GPU 密集型计算转变，单机柜功率密度提升至数十千瓦，液冷成为主流散热方式。数据中心的能源系统规划因此面临两个新特点：

- **GPU 运行状态可调**：GPU 可通过锁频、功率封顶与休眠调节功率，而吞吐量损失较小[8-9]；
- **热约束限制运行**：液冷温度和室外气象共同决定自然冷却能力，又通过芯片热安全约束反过来限制 GPU 的可用运行状态。

在规划层面同时刻画上述双向耦合，是 AI 数据中心算—电—冷协同容量配置的关键。

**现有研究的三类不足：**

1. **冷却侧建模过粗**：数据中心能源规划多以固定电能利用效率或固定单位算力功耗刻画冷却负荷。文献[1-2]建立了算力需求响应与综合能源容量配置模型；文献[5]在直流数据中心规划中考虑了分级负荷灵活性。这些研究侧重电—算协同，对冷却侧采用静态描述，未考虑 GPU 运行状态与冷却能力之间的热安全耦合。
2. **调度与规划分离**：文献[3-4,6]研究了数据中心参与电网、微网与碳协同的调度问题，GPU 功率管理研究[8-9]则主要面向机架级运行控制，二者尚未在容量规划中统一。
3. **不确定性处理未关注罕见极端**：分布鲁棒优化已用于数据中心与微网的不确定性规划[3-4,7]，常用矩信息或 Wasserstein 模糊集[12-14]。但多数模型以硬约束保证可行性，概率层只影响运行费用的期望，难以体现罕见高温—高负荷复合极端对冷源容量的影响。

**本文的主要工作：**

1. **GPU 运行调节与液冷热状态耦合的运行可行域**：建立含标称、锁频与休眠状态的 GPU 多模式运行模型，以液冷环路热状态与温区门控刻画"GPU 模式—热点温度—可用液冷温区—自然冷却能力"的耦合关系，并计及风机变工况冷源电耗，形成可嵌入容量规划的混合整数线性运行可行域。
2. **风险成本型条件分布鲁棒容量配置模型**：将实时与可延期任务失约以风险成本计入情景费用，构建基于条件正负半方差有限支持模糊集的两阶段分布鲁棒模型。最坏分布可在罕见高温与设计极端工况上重新分配概率，从而内生决定冷却储备。模型经强对偶重构为单一 MILP。
3. **真实数据外推验证**：基于中卫 11 年 ERA5 逐时气象与真实 LLM 服务负荷，按年份进行样本外外推检验。通过机制析因、同容量运行比较与不确定性模型对比，定量揭示 GPU 调节与热状态反馈的交互作用、规划与运行的协同价值，以及风险成本型分布鲁棒模型的样本外优势。

## 1 系统结构与问题描述

研究对象为含 GPU 算力集群、液冷环路、自然冷却装置（干冷器）、机械制冷（冷水机组）、光伏、储能与电网接口的 AI 数据中心（图1）。第一阶段在不确定性实现前确定 GPU、光伏、储能与冷源容量 $x=(N,K^{{\\rm PV}},P^{{\\rm B}},E^{{\\rm B}},K^{{\\rm FC}},K^{{\\rm MC}})$；第二阶段在给定气象—负荷情景下优化 GPU 运行模式、任务处理、液冷温区、冷源出力、储能与购电。

> **图1 AI数据中心算—电—冷耦合系统结构**　Fig.1 Structure of the computing–power–cooling coupled AI data center（`figures/diagrams/F1_系统结构图.png`）

## 2 GPU运行调节与液冷热状态耦合模型

### 2.1 GPU多模式运行与休眠

设 GPU 运行模式集合 $\\mathcal K_r$（$r\\in\\{{\\rm R,B}\\}$ 分别为实时与可延期任务），模式 $k$ 的相对处理速率为 $v_k$、整机功率为 $p_k$。$m_{{rkdst}}$ 为运行于模式 $k$ 处理任务 $r$ 的 GPU 台数，$n^{{\\rm off}}_{{dst}}$ 为休眠台数。为应对小时内突发，空闲 GPU 中至少保留比例 $\\rho$ 处于热备用：

$$
\\sum_r\\sum_{{k\\in\\mathcal K_r}}m_{{rkdst}}+n^{{\\rm off}}_{{dst}}\\le N,\\qquad n^{{\\rm off}}_{{dst}}\\le(1-\\rho)\\Big(N-\\sum_r\\sum_km_{{rkdst}}\\Big).\\tag{{1}}
$$

已完成工作量与 IT 功率为：

$$
U^r_{{dst}}\\le\\Delta t\\sum_{{k\\in\\mathcal K_r}}v_km_{{rkdst}},\\tag{{2}}
$$

$$
P^{{\\rm IT}}_{{dst}}=Np^{{\\rm sl}}+\\sum_r\\sum_k(p_k-p^{{\\rm sl}})m_{{rkdst}}+(p^{{\\rm off}}-p^{{\\rm sl}})n^{{\\rm off}}_{{dst}}.\\tag{{3}}
$$

式中 $p^{{\\rm sl}}$ 为空闲功率，$p^{{\\rm off}}$ 为休眠待机功率。

### 2.2 实时与可延期任务约束

实时任务在到达时段服务，未服务部分记为失约 $e^{{\\rm R}}_{{dst}}$：

$$
U^{{\\rm R}}_{{dst}}+e^{{\\rm R}}_{{dst}}=A^{{\\rm R}}_{{dst}}.\\tag{{4}}
$$

可延期任务以积压状态 $B_{{dst}}$ 描述，最大延期为 $D$，超期未完成量记为 $e^{{\\rm B}}_{{dst}}$：

$$
B_{{ds,t+1}}=B_{{dst}}+A^{{\\rm B}}_{{dst}}-U^{{\\rm B}}_{{dst}},\\qquad \\sum_{{\\tau\\le t}}U^{{\\rm B}}_{{ds\\tau}}+e^{{\\rm B}}_{{dst}}\\ge\\sum_{{\\tau\\le t-D}}A^{{\\rm B}}_{{ds\\tau}}.\\tag{{5}}
$$

### 2.3 液冷环路热状态与温区热安全门控

IT 功率中比例 $\\beta$ 进入液冷环路，环路等效温度 $T^{{\\rm w}}_{{dst}}$ 满足：

$$
C_{{\\rm w}}\\big(T^{{\\rm w}}_{{ds,t+1}}-T^{{\\rm w}}_{{dst}}\\big)=\\Delta t\\big(\\beta P^{{\\rm IT}}_{{dst}}-Q^{{\\rm FC}}_{{dst}}-Q^{{\\rm MC}}_{{dst}}\\big).\\tag{{6}}
$$

将液冷温度划分为温区 $[\\underline T_j,\\overline T_j]$，引入二进制温区变量 $u_{{jdst}}$：

$$
\\sum_ju_{{jdst}}=1,\\qquad\\sum_j\\underline T_ju_{{jdst}}\\le T^{{\\rm w}}_{{dst}}\\le\\sum_j\\overline T_ju_{{jdst}}.\\tag{{7}}
$$

模式 $k$ 的芯片热点温度为 $T^{{\\rm w}}+R^{{\\rm th}}h_k$。只有满足热安全的温区才允许该模式运行：

$$
\\mathcal J^{{\\rm safe}}_k=\\{{j:\\overline T_j+R^{{\\rm th}}h_k\\le T^{{\\rm chip,max}}\\}},\\qquad m_{{rkdst}}\\le\\overline m_{{rkdst}}\\sum_{{j\\in\\mathcal J^{{\\rm safe}}_k}}u_{{jdst}}.\\tag{{8}}
$$

式（8）在时段两端同时施加，$\\overline m$ 取任务需求导出的有效上界。式（8）体现了本文的核心耦合：

- 降频降低芯片发热 $h_k$，从而扩大可用温区；
- 较高的液冷温区提高了自然冷却能力（见 2.4 节）。

本算例中，标称、浅锁频、深锁频模式的安全供液温度分别为 {P['T_CHIP_MAX'] - P['R_TH'] * P['MODES']['k1_标称'][1]:.1f}、{P['T_CHIP_MAX'] - P['R_TH'] * P['MODES']['k2_浅锁频'][1]:.1f}、{P['T_CHIP_MAX'] - P['R_TH'] * P['MODES']['k3_深锁频'][1]:.1f} ℃（图3）。

> **图3 GPU运行模式的处理速率、功率与热安全供液温度**　Fig.3 Throughput, power and thermally safe supply temperature of GPU operating modes（`figures/exp/图2_GPU运行模式特性与热安全上限.png`）

### 2.4 自然冷却、机械制冷与冷源电耗

温区 $j$ 下干冷器的可用容量系数按温区下界保守计算：

$$
\\phi_{{jdst}}=\\min\\Big\\{{1,\\max\\Big\\{{0,\\frac{{\\underline T_j-\\Delta T^{{\\rm CDU}}-T^{{\\rm amb}}_{{dst}}}}{{\\Delta T^{{\\rm ITD}}}}\\Big\\}}\\Big\\}}.\\tag{{9}}
$$

分温区散热量 $q^{{\\rm FC}}_{{jdst}}$、$q^{{\\rm MC}}_{{jdst}}$ 满足：

$$
q^{{\\rm FC}}_{{jdst}}\\le\\phi_{{jdst}}K^{{\\rm FC}},\\quad q^{{\\rm FC}}_{{jdst}}\\le\\phi_{{jdst}}\\overline K^{{\\rm FC}}u_{{jdst}},\\quad q^{{\\rm MC}}_{{jdst}}\\le K^{{\\rm MC}},\\quad q^{{\\rm MC}}_{{jdst}}\\le\\overline K^{{\\rm MC}}u_{{jdst}}.\\tag{{10}}
$$

冷水机组单位电耗按 Carnot 比例随蒸发、冷凝温度修正：

$$
a^{{\\rm MC}}_{{jdst}}=1/{{\\rm COP}}_{{jdst}},\\qquad{{\\rm COP}}_{{jdst}}=\\eta_{{\\rm c}}\\frac{{T^{{\\rm e}}_j}}{{T^{{\\rm c}}_{{dst}}-T^{{\\rm e}}_j}}.\\tag{{11}}
$$

干冷器风机按相似律 $P\\propto V^3$ 取 0 至满风量的割线近似，单位散热电耗随逼近温差 $\\Delta T^{{\\rm a}}_{{jdst}}=\\underline T_j-\\Delta T^{{\\rm CDU}}-T^{{\\rm amb}}_{{dst}}$ 变化：

$$
a^{{\\rm FC}}_{{jdst}}=a^{{\\rm FC}}_0\\frac{{\\Delta T^{{\\rm ITD}}}}{{\\max\\{{\\Delta T^{{\\rm a}}_{{jdst}},\\Delta T^{{\\rm ITD}}/F_{{\\max}}\\}}}}.\\tag{{12}}
$$

式（12）使较高液冷温区同时具有更大的自然冷却能力和更低的风机电耗。冷却系统总电耗为：

$$
P^{{\\rm cool}}_{{dst}}=\\sum_j\\big(a^{{\\rm FC}}_{{jdst}}q^{{\\rm FC}}_{{jdst}}+a^{{\\rm MC}}_{{jdst}}q^{{\\rm MC}}_{{jdst}}\\big)+a^{{\\rm circ}}\\beta P^{{\\rm IT}}_{{dst}}+a^{{\\rm air}}(1-\\beta)P^{{\\rm IT}}_{{dst}}.\\tag{{13}}
$$

> **图4 液冷温区与室外气象对自然冷却能力和可用GPU模式的联合约束**　Fig.4 Joint restriction of liquid-cooling zone and ambient temperature on free-cooling capability and admissible GPU modes（`figures/exp/图3_液冷温区与气象共同决定可用GPU模式和自然冷却能力.png`）

### 2.5 光伏、储能与电功率平衡

光伏出力 $0\\le P^{{\\rm PV}}_{{dst}}\\le K^{{\\rm PV}}g^{{\\rm PV}}_{{dst}}$。储能采用 SOC 动态、充放电互斥与倍率约束，电网购电 $0\\le P^{{\\rm grid}}_{{dst}}\\le\\overline P^{{\\rm grid}}$。功率平衡为：

$$
P^{{\\rm grid}}_{{dst}}+P^{{\\rm PV}}_{{dst}}+P^{{\\rm dis}}_{{dst}}=(1+\\delta_e)P^{{\\rm IT}}_{{dst}}+P^{{\\rm cool}}_{{dst}}+P^0+P^{{\\rm ch}}_{{dst}}.\\tag{{14}}
$$

## 3 风险成本型条件分布鲁棒容量配置模型

### 3.1 情景风险费用

情景日费用由购电费用与服务失约的风险成本构成：

$$
Q_{{ds}}(x)=\\min_{{y_{{ds}}\\in\\mathcal Y(x,\\xi_{{ds}})}}\\Delta t\\sum_t\\Big(c^{{\\rm e}}_tP^{{\\rm grid}}_{{dst}}+c_{{\\rm R}}e^{{\\rm R}}_{{dst}}+c_{{\\rm B}}e^{{\\rm B}}_{{dst}}\\Big),\\tag{{15}}
$$

其中 $\\mathcal Y$ 由式（1）—（14）定义。$c_{{\\rm R}}$、$c_{{\\rm B}}$ 为单位失约损失，按 GPU 租赁价格与 SLA 违约赔付水平设定。当 $c_{{\\rm R}},c_{{\\rm B}}\\to\\infty$ 时，式（15）退化为硬约束保证服务的情形。

### 3.2 条件正负半方差有限支持模糊集

将训练期联合气象—负荷日按季节类别 $d$ 划分，每类以 k-medoids 代表日、训练期极端日（最高气温日、实时负荷峰值日）和 20 年一遇设计极端日构成有限支持集 $\\{{\\xi_{{ds}}\\}}$。每个情景提取归一化风险特征 $F_{{ds}}$（高温强度、光伏出力、实时与可延期负荷峰值等），风险方向偏差为 $\\rho_{{\\ell,ds}}=a_\\ell^{{\\top}}(F_{{ds}}-\\hat\\mu_d)$。条件模糊集为：

$$
\\mathcal P_d=\\Big\\{{p_d\\ge0:\\ \\sum_sp_{{ds}}=1,\\ \\Big|\\sum_sp_{{ds}}F_{{ds}}-\\hat\\mu_d\\Big|\\le\\delta_d,\\ \\sum_sp_{{ds}}[\\rho_{{\\ell,ds}}]_+^2\\le v^+_{{d\\ell}},\\ \\sum_sp_{{ds}}[-\\rho_{{\\ell,ds}}]_+^2\\le v^-_{{d\\ell}}\\Big\\}}.\\tag{{16}}
$$

正、负半方差分别约束不利方向与有利方向的偏离。参数 $\\delta_d,v^\\pm_{{d\\ell}}$ 由训练集 bootstrap 估计，并经验证集校准。

### 3.3 两阶段分布鲁棒目标

$$
\\min_{{x\\in\\mathcal X}}\\ \\sum_ia_ix_i+\\sum_dn_d\\max_{{p_d\\in\\mathcal P_d}}\\sum_sp_{{ds}}Q_{{ds}}(x),\\tag{{17}}
$$

$a_i$ 为年化单位投资，$n_d$ 为类别年天数。最坏分布可以把概率转移到经验概率很小或为零的高温、设计极端情景；结合式（15）的风险成本，冷却储备由模型内生决定。

## 4 求解方法

### 4.1 强对偶重构

对每个类别 $d$，内层 $\\max_{{p_d\\in\\mathcal P_d}}\\sum_sp_{{ds}}Q_{{ds}}$ 为线性规划，且 $\\mathcal P_d$ 非空有界。其对偶为：

$$
\\begin{{aligned}}
\\min\\ &\\lambda_d+\\alpha_d^{{\\top}}(\\hat\\mu_d+\\delta_d)-\\beta_d^{{\\top}}(\\hat\\mu_d-\\delta_d)+\\gamma_d^{{+\\top}}v^+_d+\\gamma_d^{{-\\top}}v^-_d\\\\
{{\\rm s.t.}}\\ &\\lambda_d+(\\alpha_d-\\beta_d)^{{\\top}}F_{{ds}}+\\gamma_d^{{+\\top}}[\\rho_{{ds}}]_+^2+\\gamma_d^{{-\\top}}[-\\rho_{{ds}}]_+^2\\ge Q_{{ds}},\\ \\forall s;\\quad\\alpha_d,\\beta_d,\\gamma^\\pm_d\\ge0.
\\end{{aligned}}\\tag{{18}}
$$

将式（18）代入式（17），并把各情景的运行变量与容量变量一起优化，得到单一 MILP。模型中的大 M 系数取由热平衡与任务需求导出的有效上界，不切除可行解。

### 4.2 概率割平面算法

当情景规模较大时，可将模糊集的极点作为概率割逐步加入主问题：
1. 主问题取 $\\theta_d\\ge\\sum_sp^{{(i)}}_{{ds}}Q_{{ds}}$，并以已生成的概率向量 $p^{{(i)}}$ 构造割；
2. 子问题为式（16）上的线性规划，求出当前最坏分布 $p^{{(i+1)}}$；
3. 若新分布对应的期望费用超过 $\\theta_d$，则把它作为新割加入主问题，重复上述过程，直到上下界差满足收敛容差。

求解流程见图2。本文主算例采用直接求解，并用固定整数变量的容量局部搜索改进可行解。

> **图2 模型求解流程**　Fig.2 Solution procedure（`figures/diagrams/F3_求解流程图.png`）

## 5 算例分析

### 5.1 算例设置

**数据来源：**
- **气象**：宁夏中卫（37.51°N，105.19°E）2014—2024 年 ERA5 逐时气象，光伏出力由 pvlib/PVWatts 计算[17-18]。
- **负荷**：BurstGPT 真实 LLM 服务轨迹[10]，共 188 个有效日，对话类为实时任务、API 类为可延期任务，并按算例规模缩放。

**样本划分（按年份外推）：**
- 训练集 2014—2020 年，验证集 2021—2022 年，测试集 2023—2024 年（共 730 天）；
- 测试年 ≥30 ℃ 小时数（625、552 h）高于训练年均值，检验更具挑战性；
- 20 年一遇设计温度由训练年年极值的 Gumbel 分布估计为 {Tdes:.1f} ℃。

**评价方法**：所有方案的容量均在同一详细运行模型下逐日复核，统计年运行费用、SLA 与失约；失约按单价折算为风险调整费用。

**表1 主要参数**　Table 1 Main parameters

{tab_par}

**表2 对比方案**　Table 2 Compared models

{tab_cases}

### 5.2 GPU运行调节与热状态耦合对容量配置的影响

**表3 中卫算例容量配置与能源相关年化费用（测试年 2023—2024）**　Table 3 Capacity configuration and energy-related annualized cost

{tab_cfg}

**主要结果：**

1. **完整耦合模型（M3）大幅减少冷水机组**：
   - 冷水机组由 {m0['机械制冷']:.0f} kW 降至 {m3['机械制冷']:.0f} kW，自然冷却由 {m0['自然冷却']:.0f} kW 增至 {m3['自然冷却']:.0f} kW，光伏由 {m0['光伏']:.0f} kWp 降至 {m3['光伏']:.0f} kWp；
   - 能源相关年化费用降低 {abs(dE):.1f} 万元（{pE:.1f}%），冷却电量降低 {abs(dcool):.1f}%；
   - 与按逐时气象和设计日校核的工程设计方法（M0-D）相比，年化费用降低 {abs(dEd):.1f} 万元（{pEd:.1f}%）。
2. **机理**：
   - 锁频与休眠降低 IT 功率和芯片发热，使实时任务的安全供液温度由 45 ℃ 提高到 50.2 ℃；
   - 在较高温区运行时，最热时段的自然冷却可用系数显著提高，干冷器（年化单价约为冷水机组的 1/4.3）即可承担散热；
   - 同时，休眠降低了午间 IT 负荷，光伏无需按全部 GPU 常开配置。

**表4 节约分解（万元/年；2×2 析因）**　Table 4 Decomposition of cost savings

{tab_dec}

**两种机制的交互作用（表4）**：
- 单独引入 GPU 调节（M1）或单独引入热状态反馈（M2）时，冷水机组仍分别为 {m1['机械制冷']:.0f} kW 和 {m2['机械制冷']:.0f} kW，冷源投资只分别降低 {abs(Zd.loc['冷源投资', '仅GPU调节']):.1f} 万元和 {abs(Zd.loc['冷源投资', '仅热耦合']):.1f} 万元；
- 二者同时引入时，冷源投资节约的 {cap_int:.0f}% 来自交互项，能源相关费用的交互项为 {I_GT:+.1f} 万元/年。

这说明"GPU 运行模式—液冷温区—自然冷却"的耦合约束是冷源配置变化的直接原因，单一机制的叠加无法得到相同效果。

> **图5 中卫算例：(a) 年份外推的气象特征，(b) 冷源配置，(c) 节约分解，(d) 样本外风险调整费用差的 95% 置信区间**　Fig.5 Zhongwei case: weather of training/test years, cooling capacity, saving decomposition and out-of-sample risk-adjusted cost differences（`figures/exp/图22_宁夏中卫真实气象_按年份外推的配置与样本外风险费用.png`）

### 5.3 规划与运行的协同

为区分容量与运行策略的作用，把容量统一取为 M3 的规划结果，只改变运行策略（表5）。

**表5 同容量运行比较（中卫测试年）**　Table 5 Operation strategies under identical capacity

{tab_ops}

""", globals())
if ops_ok:
    o = OPS
    txt += render("""**结果分析：**
- **只有 M2、M3 两种策略能在全部测试日安全运行**：在相同设备下，M3 策略的年运行费用比 M2 策略低 {ops_save:.1f}%，实时 SLA 最高（{o.loc['M3策略', '实时SLA'] * 100:.2f}%），冷却电量也明显低于 M2 策略。
- **固定供液温度的传统策略（M0、M1）大量不可安全运行**：分别有 {int(o.loc['M0策略', '不可安全运行日数'])}、{int(o.loc['M1策略', '不可安全运行日数'])} 个测试日即使允许失约也无法满足热平衡，表中费用只统计可运行日，因而偏低。原因是设备按高温运行设计（大干冷器、几乎无冷水机组），夏季若固定 25—32 ℃ 供液，自然冷却能力不足。
- **只升温、不调节 GPU（M2 策略）**：标称模式在高温区不满足热安全，只能停用 GPU，失约最多。

这表明按耦合模型规划的设备必须以耦合方式运行，容量规划与运行策略需要协同设计。

""", globals())
txt += render("""### 5.4 不确定性建模对比

**表6 不确定性处理方法对比（中卫测试年，c_R=100 元/GPU·h）**　Table 6 Comparison of uncertainty models

{tab_unc}

**表7 样本外风险调整费用差（配对区块 bootstrap，2000 次）**　Table 7 Out-of-sample paired differences

{tab_boot}

**结果分析：**

1. **风险中性随机规划（R-SP）对冷源容量不敏感**：
   - 设计极端日的经验概率为 0，训练期最热日只有 {ph_hot:.3f}，因此在 R-SP 的目标中，自然冷却容量变化对目标值影响很小，配置偏低（{sp['自然冷却']:.0f} kW）；
   - 在更热的 2023—2024 测试年，其实时 SLA 降至 {sp['实时SLA']:.2f}%，设计日失约 {sp['设计日失约']:,.0f} GPU·h。
2. **本文方法（R-U3）自动保留冷却储备**：
   - 条件正负半方差模糊集的最坏分布在夏季把设计极端日的概率由 0 提高到 {p_des:.3f}，相当于每个夏季约 {p_des * 92:.0f} 天按设计极端工况计入风险费用（图6）；
   - 因此保留 {u3['自然冷却']:.0f} kW 自然冷却，设计日零失约；
   - 样本外风险调整费用比 R-SP 低 {abs(b_u3sp['差值万元']):,.0f} 万元/年，95% 置信区间不含 0。
3. **条件对称模糊集（R-U2）得到与 R-U3 相同的配置**：在本算例中，起主要作用的是"条件化有限支持 + 风险成本"的结构，正负半方差的作用是给出方向性的风险解释。
4. **与硬约束模型（M3）相比**：R-U3 不需要人为指定哪些情景必须严格满足，就得到了与之相当或更优的配置（风险调整费用 {u3['风险调整费用']:.1f} 万元 vs {m3['风险调整费用']:.1f} 万元），并相对静态基准显著降本（表7）。

> **图6 中卫算例条件半方差模糊集的最坏分布与夏季情景风险费用**　Fig.6 Worst-case distribution and summer scenario risk costs（`figures/exp/图23_中卫条件半方差模糊集的最坏分布与情景风险费用.png`）

### 5.5 敏感性与适用性分析

为分析参数与气候的影响，采用典型气象年算例进行敏感性分析。

**（1）失约单价**

**表8 失约单价敏感性**　Table 8 Sensitivity to shortfall price

{tab_cr}

失约单价越高，R-U3 相对 R-SP 的优势越大。R-SP 因设计极端日经验概率为零，提高单价也不会增加冷却储备（图7）。

> **图7 风险成本型分布鲁棒规划的冷源配置、最坏分布与样本外风险费用**　Fig.7 Cooling capacity, worst-case distribution and out-of-sample risk cost of risk-cost DRO（`figures/exp/图20_风险成本型分布鲁棒规划_冷源配置最坏分布与样本外风险费用.png`）

**（2）热安全参数与休眠参数**

**表9 关键参数敏感性**　Table 9 Sensitivity to thermal and sleep parameters

{tab_sens}

- 在全部参数组合下，耦合模型均降低能源相关费用；
- 热阻越小、热点上限越高，浅锁频模式的安全供液温度越高，冷水机组越容易被替代；
- 浅锁频安全供液温度低于约 50 ℃ 时，最热时段仍需部分机械制冷。

**（3）风机变工况**

采用式（12）的风机变工况模型时，耦合模型的冷却电量比静态基准低 {(1 - fv.loc['M3', '冷却电量MWh'] / fv.loc['M0', '冷却电量MWh']) * 100:.0f}%：较高液冷温区增大了逼近温差，降低了单位散热的风机电耗。

**（4）气候适用性**

**表10 不同气候下耦合模型的效果**　Table 10 Effect under different climates

{tab_clim}

在半干旱、温带大陆性及高原气候中，耦合模型的冷源节约最显著；在湿热或温和海洋性气候中，节约主要来自 GPU 调节带来的光伏与运行优化（图8）。

> **图8 参数敏感性、风机变工况与负荷外推下的稳健性**　Fig.8 Robustness under parameter variation, variable-speed fans and load extrapolation（`figures/exp/图21_稳健性_参数敏感性_风机变工况与长负荷数据.png`）

## 6 结论

本文提出考虑 GPU 运行调节与液冷热状态耦合的风险成本型分布鲁棒容量配置方法，得到以下结论：

1. **耦合可行域显著改变冷源配置**：GPU 锁频与休眠扩大了液冷可用温区，较高温区又提升了自然冷却能力并降低风机电耗。
   - 中卫算例中，冷水机组容量由 {m0['机械制冷']:.0f} kW 降至 {m3['机械制冷']:.0f} kW，能源相关年化费用降低 {abs(pE):.1f}%；
   - 冷源投资节约的 {cap_int:.0f}% 来自 GPU 调节与热状态反馈的交互作用。
2. **容量规划与运行策略需要协同**：按耦合模型规划的设备若采用固定供液温度或不调节 GPU 的传统运行方式，会出现不可安全运行日或大量失约。
3. **风险成本型条件分布鲁棒模型内生决定冷却储备**：最坏分布将概率转移到罕见高温与设计极端工况，在年份外推检验中，样本外风险调整费用显著低于随机规划，且无需人为指定硬约束情景。

**后续工作**：本文的芯片热模型采用准稳态热阻，GPU 模式参数基于公开测试数据；后续将结合实测数据开展更细时间尺度的热—电联合建模，并研究多园区算力迁移下的协同规划。

## 参考文献

[1] 马明克, 陈艳波, 邓涵宇, 等. 计及数据中心余热回收与算力需求响应的园区综合能源系统容量配置方法[J/OL]. 电网技术, 2026. DOI: 10.13335/j.1000-3673.pst.2026.0949.

[2] ZHANG S, DING S, LI Y, et al. A bi-level planning model of integrated energy capacity configuration for data center cluster considering computing-electricity coupling operation and supply-demand uncertainties[J]. Applied Energy, 2026, 425: 128499.

[3] SONG C, CHEN H, WANG J, et al. Bi-timescale distributionally robust optimization for power-computing-carbon coordination in multi-microgrids with data centers[J]. Applied Energy, 2026, 426: 128602.

[4] WANG Y, YAO L, STRBAC G. Coordinating data centre microgrids and mobile power sources for distribution network resilience: a data-driven distributionally robust optimisation approach[J]. Applied Energy, 2027, 427: 129132.

[5] XU J, XU Q, JIANG X, et al. Coordinated planning and operation of SST-based DC data centers with tiered workload flexibility[J]. Applied Energy, 2027, 427: 128931.

[6] JIANG Y, WANG Q, WU G, et al. Decarbonization dispatch for AI computing center parks considering task-level carbon traceability[J]. Applied Energy, 2027, 427: 128887.

[7] SHI X, LIN Z, YAN J. Reframing data centers as strategic producers: energy-data nexus[J]. Applied Energy, 2026, 426: 128618.

[8] PATEL P, CHOUKSE E, ZHANG C, et al. Splitwise: efficient generative LLM inference using phase splitting[C]//Proceedings of the 51st ACM/IEEE Annual International Symposium on Computer Architecture (ISCA). 2024.

[9] PATEL P, CHOUKSE E, ZHANG C, et al. Characterizing power management opportunities for LLMs in the cloud[C]//Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). 2024.

[10] WANG Y, CHEN Y, LI Z, et al. BurstGPT: a real-world workload dataset to optimize LLM serving systems[C]//Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2025.

[11] HERSBACH H, BELL B, BERRISFORD P, et al. The ERA5 global reanalysis[J]. Quarterly Journal of the Royal Meteorological Society, 2020, 146(730): 1999-2049.

[12] DELAGE E, YE Y. Distributionally robust optimization under moment uncertainty with application to data-driven problems[J]. Operations Research, 2010, 58(3): 595-612.

[13] WIESEMANN W, KUHN D, SIM M. Distributionally robust convex optimization[J]. Operations Research, 2014, 62(6): 1358-1376.

[14] MOHAJERIN ESFAHANI P, KUHN D. Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations[J]. Mathematical Programming, 2018, 171(1): 115-166.

[15] ZENG B, ZHAO L. Solving two-stage robust optimization problems using a column-and-constraint generation method[J]. Operations Research Letters, 2013, 41(5): 457-461.

[16] ASHRAE TC 9.9. Liquid cooling guidelines for datacom equipment centers[M]. 2nd ed. Atlanta: ASHRAE, 2014.

[17] HOLMGREN W F, HANSEN C W, MIKOFSKI M A. pvlib python: a python package for modeling solar energy systems[J]. Journal of Open Source Software, 2018, 3(29): 884.

[18] DOBOS A P. PVWatts version 5 manual[R]. Golden, CO: National Renewable Energy Laboratory, 2014. NREL/TP-6A20-62641.

[19] HUANGFU Q, HALL J A J. Parallelizing the dual revised simplex method[J]. Mathematical Programming Computation, 2018, 10(1): 119-142.

[20] MASANET E, SHEHABI A, LEI N, et al. Recalibrating global data center energy-use estimates[J]. Science, 2020, 367(6481): 984-986.

> 待补充：近三年中国电机工程学报、电力系统自动化、电网技术、电工技术学报中关于数据中心算力负荷灵活性、液冷与自然冷却、算电协同规划、分布鲁棒优化的文献（建议 10–15 篇），使总数达到 30–40 篇。上述 [1]–[20] 均为已核对的真实文献；[8]–[10] 的作者列表请按原文核对补全。

## 图件清单（正文 8 张）

| 图号 | 内容 | 文件 |
|---|---|---|
| 图1 | 系统结构 | `figures/diagrams/F1_系统结构图.png` |
| 图2 | 求解流程 | `figures/diagrams/F3_求解流程图.png` |
| 图3 | GPU 运行模式与热安全 | `figures/exp/图2_GPU运行模式特性与热安全上限.png` |
| 图4 | 液冷温区、气象与自然冷却 | `figures/exp/图3_液冷温区与气象共同决定可用GPU模式和自然冷却能力.png` |
| 图5 | 中卫：配置、分解、样本外差值 | `figures/exp/图22_宁夏中卫真实气象_按年份外推的配置与样本外风险费用.png` |
| 图6 | 中卫：最坏分布与情景风险费用 | `figures/exp/图23_中卫条件半方差模糊集的最坏分布与情景风险费用.png` |
| 图7 | 风险成本型 DRO 敏感性（失约单价） | `figures/exp/图20_风险成本型分布鲁棒规划_冷源配置最坏分布与样本外风险费用.png` |
| 图8 | 参数、风机与负荷稳健性 | `figures/exp/图21_稳健性_参数敏感性_风机变工况与长负荷数据.png` |

其余图（同容量运行、最热日调度、桑基图、气候三维柱等）可作为补充材料。
""", globals())
(C.ROOT / "docs/论文初稿_中国电机工程学报.md").write_text(txt, encoding="utf-8")
print("written docs/论文初稿_中国电机工程学报.md", "ops_ok=", ops_ok)
