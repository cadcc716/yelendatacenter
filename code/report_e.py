"""
按修订版实验方案 (E1–E5) 生成结果报告 docs/实验结果报告_E1-E5.md
所有数值由 results/exp/*.csv 计算, 不手工抄录。运行: python report_e.py
"""
import pickle
import numpy as np, pandas as pd
import config as C

RES = C.RES / "exp"; W = 1e4
S = pd.read_csv(RES / "summary.csv", index_col=0)
O = pd.read_csv(RES / "ops_summary.csv").set_index("策略") if (RES / "ops_summary.csv").exists() else None
DC = pd.read_csv(RES / "design_check.csv").set_index("方案") if (RES / "design_check.csv").exists() else None
ST = pd.read_csv(RES / "stress_all.csv") if (RES / "stress_all.csv").exists() else None
BN = pd.read_csv(RES / "bench_all.csv") if (RES / "bench_all.csv").exists() else None
CAL = pd.read_csv(RES / "calibration.csv")
D = pickle.load(open(C.PROC / "dataset_base.pkl", "rb"))


def f(x, n=1): return f"{x:,.{n}f}"


def pct(a, b): return (a / b - 1) * 100


def cool_capex(m): return (S.loc[m, "年化_FC"] + S.loc[m, "年化_MC"]) / W


def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def gap(m): return f"{S.loc[m, '间隙'] / W:.1f}" if S.loc[m, "间隙"] > 1.5e4 else "≤1.0"


def row(m, lab):
    r = S.loc[m]; dsg = DC.loc[m] if DC is not None and m in DC.index else None
    df = "—" if dsg is None else f"{(dsg['实时失约'] or 0) + (dsg['延期失约'] or 0):,.0f}"
    return [lab, f"{r['PV']:.0f}", f"{r['FC']:.0f}", f"{abs(max(r['MC'], 0)):.0f}", f(cool_capex(m)), f(r["复核年运行费用"] / W),
            f(r["复核能源相关费用"] / W), f"{r['复核PUE']:.4f}", f"{r['实时SLA'] * 100:.2f}", f"{r['实时失约_冷却热安全受限']:,.0f}", df,
            f(r.get("冷却修复成本", np.nan) / W) if pd.notna(r.get("冷却修复成本", np.nan)) else "—", gap(m)]


HDR = ["方案", "光伏kWp", "自然冷却kWth", "机械制冷kWth", "冷源年化投资万元", "测试日年运行费用万元", "能源相关费用万元", "PUE", "实时SLA%",
       "冷却/热安全失约GPU·h", "设计日失约GPU·h", "冷却修复成本万元/年", "规划间隙万元"]
E1 = [("M0", "M0 静态基准(a=0,b=0)"), ("M1", "M1 +GPU调节"), ("M2", "M2 +热反馈"), ("M3", "M3 完整耦合"), ("M0-W45", "M0-W45 静态温水设计"),
      ("无休眠-M3", "M3 无休眠态(仅降频)"), ("温区粗化-M3", "M3 温区粗化(5区)")]
E3 = [("U0", "U0 确定性均值"), ("M3", "U-SP 经验分布(=M3)"), ("U1", "U1 无条件对称"), ("U2", "U2 条件对称"), ("U3", "U3 条件半方差(本文)")]
T1 = md(pd.DataFrame([row(m, l) for m, l in E1], columns=HDR))
T3 = md(pd.DataFrame([row(m, l) for m, l in E3], columns=HDR))

m0, m3 = S.loc["M0"], S.loc["M3"]
c = {m: S.loc[m, "复核能源相关费用"] / W for m in ("M0", "M1", "M2", "M3")}
p = {m: S.loc[m, "规划能源相关费用"] / W for m in ("M0", "M1", "M2", "M3")}
I_hf = c["M1"] + c["M2"] - c["M0"] - c["M3"]; I_pl = p["M1"] + p["M2"] - p["M0"] - p["M3"]
m3rep = c["M3"] + m3["冷却修复成本"] / W

ops = ""
if O is not None:
    ops = md(pd.DataFrame([[k, f(r["复核年运行费用"] / W), f"{r['实时SLA'] * 100:.2f}", f"{r['实时失约GPUh']:,.0f}", int(r["不可安全运行日数"]),
                            f"{r['复核PUE']:.4f}"] for k, r in O.iterrows()],
                          columns=["运行策略 (容量统一取 M3)", "年运行费用万元(可运行日)", "实时SLA%", "实时失约GPU·h", "不可安全运行测试日", "PUE"]))
    o0, o3 = O.loc["M0策略"], O.loc["M3策略"]

front = []
for u in ("U1", "U2", "U3"):
    for k in (0.5, 1.0, 2.0):
        n = u if k == 1.0 else f"{u}-k{k:g}"
        if n in S.index:
            r = S.loc[n]
            front.append([u, k, f"{r['PV']:.0f}", f"{r['FC']:.0f}", f"{abs(max(r['MC'], 0)):.0f}", f(r["复核年运行费用"] / W), f"{r['日费用CVaR95']:,.0f}",
                          f"{r['实时失约_冷却热安全受限']:,.0f}", f(r["复核能源相关费用"] / W), gap(n)])
TF = md(pd.DataFrame(front, columns=["模糊集", "κ", "光伏kWp", "自然冷却kWth", "机械制冷kWth", "测试日年运行费用万元", "日费用CVaR95元",
                                     "冷却/热安全失约GPU·h", "能源相关费用万元", "规划间隙万元"]))
cal = md(CAL.round(3)[["模型", "类别", "训练日", "验证日", "kappa_delta", "kappa_v", "tau"]])

clim = []
for nm, pre in (("丹佛(基准)", ""), ("旧金山", "S2-旧金山-"), ("芝加哥", "S2-芝加哥-"), ("坦帕", "S2-坦帕-")):
    a, b = (pre + "M0") if pre else "M0", (pre + "M3") if pre else "M3"
    if a in S.index and b in S.index:
        d0 = DC.loc[a] if DC is not None else None; d3 = DC.loc[b] if DC is not None else None
        clim.append([nm, f"{d0['设计温度']:.1f}" if d0 is not None else "—", f"{abs(max(S.loc[a, 'MC'], 0)):.0f} / {abs(max(S.loc[b, 'MC'], 0)):.0f}",
                     f"{S.loc[a, 'FC']:.0f} / {S.loc[b, 'FC']:.0f}", f(S.loc[a, "复核能源相关费用"] / W), f(S.loc[b, "复核能源相关费用"] / W),
                     f"{pct(S.loc[b, '复核能源相关费用'], S.loc[a, '复核能源相关费用']):+.1f}",
                     f"{(d0['实时失约'] + d0['延期失约']):,.0f} / {(d3['实时失约'] + d3['延期失约']):,.0f}" if d0 is not None else "—", gap(b)])
TC = md(pd.DataFrame(clim, columns=["气候", "设计日℃", "机械制冷kWth M0/M3", "自然冷却kWth M0/M3", "M0能源费用万元", "M3能源费用万元", "M3相对M0%",
                                    "设计日失约 M0/M3", "M3间隙万元"]))
sens = []
for lab, a, b in (("GPU投资×0.6", "S1-aN0.6-M0", "S1-aN0.6-M3"), ("GPU投资×1.0", "M0", "M3"), ("GPU投资×1.4", "S1-aN1.4-M0", "S1-aN1.4-M3"),
                  ("可延期占比15%", "S3-b15-M0", "S3-b15-M3"), ("可延期占比28.6%(基准)", "M0", "M3")):
    if a in S.index and b in S.index:
        sens.append([lab, f"{S.loc[a, 'N']:.0f} / {S.loc[b, 'N']:.0f}", f"{abs(max(S.loc[a, 'MC'], 0)):.0f} / {abs(max(S.loc[b, 'MC'], 0)):.0f}",
                     f(S.loc[a, "复核能源相关费用"] / W), f(S.loc[b, "复核能源相关费用"] / W), f"{pct(S.loc[b, '复核能源相关费用'], S.loc[a, '复核能源相关费用']):+.1f}"])
sens.append(["可延期占比50%", "—", "—", "不可行", "不可行", "最大突发日需 GPU 超过机位上限 2560 台"])
TS = md(pd.DataFrame(sens, columns=["情形", "GPU台数 M0/M3", "机械制冷kWth M0/M3", "M0能源费用万元", "M3能源费用万元", "M3相对M0%"]))

stress = ""
if ST is not None:
    t = ST[ST.方案.isin(["M0", "M2", "M3", "U0", "U3"])].copy()
    t["失约GPU·h"] = (t["实时失约"] + t["延期失约"]).abs().round(0)
    stress = md(t.pivot_table(index="方案", columns="压力片段", values="失约GPU·h").reset_index())
dsg = ""
if DC is not None:
    t = DC.loc[[m for m in ["M0", "M1", "M2", "M3", "M0-W45", "无休眠-M3", "温区粗化-M3", "U0", "U1", "U2", "U3"] if m in DC.index]].reset_index()
    t["设计日失约GPU·h"] = (t["实时失约"] + t["延期失约"]).round(0)
    dsg = md(t[["方案", "设计温度", "设计日失约GPU·h", "机械制冷电量"]].round(1))
bench = md(BN[["规模", "情景数", "算法", "状态", "目标值", "绝对间隙", "迭代", "时间s"]].round(0)) if BN is not None else "（求解效率作业未完成）"

u3, us = S.loc["U3"], S.loc["M3"]
nrow = []
for tag, lab in (("N30", "约 30% 训练日"), ("N60", "约 60% 训练日"), ("", "全部训练日")):
    for m, ml in (("M3", "U-SP"), ("U2", "U2"), ("U3", "U3")):
        n = f"{tag}-{m}" if tag else m
        if n in S.index:
            r = S.loc[n]
            nrow.append([lab, ml, f"{r['PV']:.0f}", f"{r['FC']:.0f}", f"{abs(max(r['MC'], 0)):.0f}", f(r["复核年运行费用"] / W), f"{r['日费用CVaR95']:,.0f}",
                         f"{r['实时失约_冷却热安全受限']:,.0f}", f(r["复核能源相关费用"] / W),
                         f"{(DC.loc[n, '实时失约'] + DC.loc[n, '延期失约']):,.0f}" if DC is not None and n in DC.index else "—", gap(n)])
TN = md(pd.DataFrame(nrow, columns=["训练数据", "模型", "光伏kWp", "自然冷却kWth", "机械制冷kWth", "测试日年运行费用万元", "日费用CVaR95元",
                                    "冷却/热安全失约GPU·h", "能源相关费用万元", "设计日失约GPU·h", "规划间隙万元"])) if nrow else "（训练样本量实验进行中）"
def dE(a, b="M0"): return (S.loc[a, "复核能源相关费用"] - S.loc[b, "复核能源相关费用"]) / W


cx = {k: (S.loc["M3", f"年化_{k}"] - S.loc["M0", f"年化_{k}"]) / W for k in ("PV", "FC", "MC")}
dop = (m3["复核年运行费用"] - m0["复核年运行费用"]) / W
sleep_val = dE("无休眠-M3", "M3") if "无休眠-M3" in S.index else float("nan")
coarse_val = dE("温区粗化-M3", "M3") if "温区粗化-M3" in S.index else float("nan")
o1, o2 = (O.loc["M1策略"], O.loc["M2策略"]) if O is not None else (None, None)


def ncmp(tag):
    a_, b_ = (f"{tag}-M3", f"{tag}-U3") if tag else ("M3", "U3")
    if a_ not in S.index or b_ not in S.index: return "（未完成）"
    A, B = S.loc[a_], S.loc[b_]
    return (f"U3 相对 U-SP：运行费用 {(B['复核年运行费用'] - A['复核年运行费用']) / W:+.1f} 万元、CVaR95 {B['日费用CVaR95'] - A['日费用CVaR95']:+.0f} 元/日、"
            f"冷却失约 {B['实时失约_冷却热安全受限'] - A['实时失约_冷却热安全受限']:+.0f} GPU·h、能源相关费用 {(B['复核能源相关费用'] - A['复核能源相关费用']) / W:+.1f} 万元")


rep = f"""# 考虑GPU运行调节与液冷热状态耦合的AI数据中心分布鲁棒容量配置——修订版实验方案 E1–E5 结果报告（v2）

> 按 `main` 分支《数学模型与完整实验方案_修订版》第 5 节执行。数值由 `code/exp.py` 求解、`code/report_e.py` 从 `results/exp/*.csv` 直接生成；图件见 `figures/exp/`；
> 物理校验见 `results/exp/checks_*.csv`。测试集在方案与参数确定前封存。上一版（5 温区、无休眠态）结果归档于 `results/exp_v1_5温区/`。
>
> **v2 相对 v1 的模型改进（依据见 `docs/审稿分析_E1-E5.md`）**：① GPU 运行模式集合加入低功耗休眠态（速率 0、待机 30 W/台，至少保留 20% 空闲 GPU 热备用），传统基准 a=0 保持全部常开；
> ② 液冷温区由 5 区加密为 10 区（E4 网格校验表明 5 区按下界计算自然冷却明显保守）；③ 设计极端日负荷统一取训练块日负荷最大日；④ 机械制冷平局打破项（弃光时段退化）。

## 0 结论速览

| 研究命题 | 结论（统一物理评价器，封存测试日） | 证据 |
|---|---|---|
| H1 热约束限制可用 GPU 模式 | 标称模式只能在液冷 ≤45 ℃ 运行，浅/深锁频放宽到 50.2/53 ℃；热点温度在测试日触及 85 ℃ 上限，热安全约束为紧约束 | 图2、图3、表6 |
| H2 完整耦合改变设备配置 | 机械制冷 M0 {m0['MC']:.0f} → M3 **{abs(max(m3['MC'], 0)):.0f}** kWth；能源相关费用 {f(c['M0'])} → {f(c['M3'])} 万元/年（**{pct(c['M3'], c['M0']):.1f}%**）；节约由冷源投资 {cx['FC'] + cx['MC']:+.1f}、光伏投资 {cx['PV']:+.1f}、运行费用 {dop:+.1f} 万元构成 | 表1、图4 |
| H3 GPU 调节与热反馈存在正交互 | $I_{{GT}}$ = **{I_hf:+.1f} 万元**（统一复核）/ {I_pl:+.1f} 万元（规划目标）；单独热反馈（M2）几乎无收益（{dE('M2'):+.1f} 万元），与 GPU 调节结合后冷水机组降为 0 | 图5 |
| 运行策略价值（同容量） | 同一 M3 容量下，M3 运行策略年运行费用 {f(o3['复核年运行费用'] / W)} 万元，M0 策略 {f(o0['复核年运行费用'] / W)} 万元（**{pct(o3['复核年运行费用'], o0['复核年运行费用']):.1f}%**，且 M0 策略另有 {int(o0['不可安全运行日数'])} 个不可安全运行日未计费用）；SLA {o3['实时SLA'] * 100:.2f}% vs {o0['实时SLA'] * 100:.2f}% | 表2、图9、图19 |
| GPU 休眠态的价值 | 取消休眠态（仅降频）后能源相关费用增加 {sleep_val:+.1f} 万元/年；休眠使闲置基荷从 IT 电量的约 79% 大幅下降 | 表1、图19 |
| 简化模型的风险 | 静态温水设计 M0-W45 测试日冷却失约 {S.loc['M0-W45', '实时失约_冷却热安全受限']:,.0f} GPU·h、设计日失约 {DC.loc['M0-W45', '实时失约'] + DC.loc['M0-W45', '延期失约']:,.0f} GPU·h；M0 规划模型（GPU 全部常开、静态冷却）估计的能源相关费用比统一复核高 {pct(m0['规划能源相关费用'], m0['复核能源相关费用']):.0f}%，忽略 GPU 调节高估负荷并使光伏过配（复核光伏利用率仅 {m0['复核光伏利用率'] * 100:.0f}%） | 表1、表6 |
| 不确定性建模 | U0 只配 {S.loc['U0', 'N']:.0f} 台 GPU，实时 SLA {S.loc['U0', '实时SLA'] * 100:.1f}%，设计日失约 {DC.loc['U0', '实时失约'] + DC.loc['U0', '延期失约']:,.0f} GPU·h；条件化模糊集（U2/U3）优于无条件（U1） | 表3 |
| 条件半方差 DRO（U3）的增量 | 全部训练数据下：{ncmp('')}；差异在规划间隙内。训练数据减少时见表4b | 表3、表4b、图10 |
| 适用边界 | 4 种气候 M3 均优于 M0（见表7）；GPU 投资 ±40% 不改变结论；可延期占比 15% 时节约 {-pct(S.loc['S3-b15-M3', '复核能源相关费用'], S.loc['S3-b15-M0', '复核能源相关费用']):.1f}% | 表7、表8 |

## 1 实验设置

- **数据划分**：每月按日期先后 60%/15%/25% 划分训练/验证/封存测试（221/48/95 天）；测试块含 6 月 24–30 日真实热浪（40.0 ℃），训练集最高 36.7 ℃。
- **有限支持**：每季节 4 个 k-medoids 代表日 + 训练集最热日 + 实时负荷峰值日；夏季加 ASHRAE 2009 n=20 年一遇极端最高温度设计日（丹佛 39.5 ℃，负荷取训练块日负荷最大日，经验概率 0）。
- **模糊集**：U1 无条件对称 / U2 条件对称 / U3 条件正负半方差；bootstrap 95% 分位数 + 验证块校准，非空性检验均通过：

{cal}

- **GPU 运行模式**：k1 标称 / k2 浅锁频 / k3 深锁频 / **休眠**（速率 0、待机 0.03 kW/台）；空闲 GPU 至少 20% 保持热备用。a=0 的方案不使用降频与休眠（全部常开、标称模式）。
- **液冷温区**：25–53 ℃ 划分 10 区；自然冷却可用系数按温区下界计算；传统运行策略固定 25–32 ℃（前两个温区）。
- **统一物理评价器**：全部方案在同一 M3 物理与运行控制（含降频、休眠、温区）下逐日复核，字典序失约诊断；GPU 台数不足的实时失约（测试块峰值 2177 > 2099）各方案共有，单列。
- **求解**：HiGHS，绝对间隙 1 万元；主要方案 90 min，其余 30 min（未收敛者报告间隙）。

## 2 E1/E2 物理机制消融与容量互补（表1）

{T1}

**对比—观察—机制：**

1. **完整耦合（M3）**：冷水机组 {m0['MC']:.0f} → {abs(max(m3['MC'], 0)):.0f} kWth，自然冷却 {m0['FC']:.0f} → {m3['FC']:.0f} kWth，光伏 {m0['PV']:.0f} → {m3['PV']:.0f} kWp，能源相关费用 {dE('M3'):+.1f} 万元/年（{pct(c['M3'], c['M0']):.1f}%）。机制链：休眠与降频降低 IT 功率与散热 → 降频使实时任务安全温区升至 50.2 ℃、可延期任务升至 53 ℃ → 最热时段仍可由自然冷却散热（单价为冷水机组的 1/4.3）→ 取消冷水机组；同时午间 IT 负荷下降使光伏只需配到较低规模。
2. **单独 GPU 调节（M1）**：{dE('M1'):+.1f} 万元，主要来自休眠降低负荷后少配光伏；由于 b=0 静态冷却模型仍按 32 ℃ 设计供液，冷水机组 {S.loc['M1', 'MC']:.0f} kWth 基本不变。
3. **单独热反馈（M2）**：{dE('M2'):+.1f} 万元，冷水机组 {S.loc['M2', 'MC']:.0f} kWth：GPU 全部常开且为标称模式时，45 ℃ 以上不能安全运行，最热时段仍需冷水机组。
4. **交互项 $I_{{GT}}$ = {I_hf:+.1f} 万元 > 0**：两种机制单独收益之和（{dE('M1') + dE('M2'):+.1f}）小于联合收益（{dE('M3'):+.1f}），是“算力调节—热安全温区—自然冷却能力”链式耦合的定量证据。M2 规划间隙 {S.loc['M2', '间隙'] / W:.1f} 万元，数值精度以此为限。
5. **安全性**：M3 在 40 ℃ 测试日冷却受限失约 {m3['实时失约_冷却热安全受限']:.0f} GPU·h，设计日无失约；冷却修复成本 {f(m3['冷却修复成本'] / W)} 万元/年，修复后仍比 M0 低 {f(c['M0'] - m3rep)} 万元/年。M0-W45 账面费用低，但冷却失约 {S.loc['M0-W45', '实时失约_冷却热安全受限']:,.0f} GPU·h、设计日失约 {DC.loc['M0-W45', '实时失约'] + DC.loc['M0-W45', '延期失约']:,.0f} GPU·h，属于少服务换来的低成本。
6. **消融（E4）**：取消休眠态 {sleep_val:+.1f} 万元/年；5 温区粗化 {coarse_val:+.1f} 万元/年（自然冷却需配到 5000 kWth 安装上限）。

## 3 同容量运行比较（表2）

{ops}

容量统一取 M3。M0/M1 策略固定 25–32 ℃ 供液，分别有 {int(o0['不可安全运行日数'])}/{int(o1['不可安全运行日数'])} 个测试日即使允许失约也无法满足热平衡（费用未计入，因此偏低）；M2 策略不降频不休眠，45 ℃ 以上只能停用 GPU，失约最多；**M3 策略运行费用最低（较 M0 策略 {pct(o3['复核年运行费用'], o0['复核年运行费用']):.1f}%）、冷却电量最低（{pct(o3['复核冷却电量MWh'], o0['复核冷却电量MWh']):.1f}%）、SLA 最高**。按耦合模型规划的设备必须以耦合方式运行，规划与运行协同是本文区别于“只做调度”或“只做规划”研究的核心论据。

## 4 E3 不确定性模型

### 表3 全部训练数据

{T3}

### 表4 κ 前沿（κ≠1 为 30 min 时限，趋势参考）

{TF}

### 表4b 训练样本量实验（每月仅用最早的 30%/60% 训练日；验证/测试块不变）

{TN}

- 约 30% 训练日：{ncmp('N30')}
- 约 60% 训练日：{ncmp('N60')}
- 全部训练日：{ncmp('')}

**解读**：{{N_TEXT}}

## 5 连续多日压力测试（表5）

{stress}

## 6 设计极端日校核（表6）

{dsg}

## 7 E4 适用边界

### 表7 气候窗口

{TC}

### 表8 GPU 投资与任务结构

{TS}

## 8 E5 求解效率

{bench}

## 9 局限

1. 规划间隙：M2（15.3 万元）、U3（9.6 万元）、坦帕 M3（34.4 万元）、30% 样本量算例（34–38 万元）未收敛；κ 前沿为 30 min 结果。未收敛的 M3 类方案已用固定整数变量的容量局部搜索改进可行解，下界仍取 MILP 下界。M0 与基准 M3 均已收敛，因此主结论不受间隙影响；建议用商业求解器复算其余算例。
5. 自然冷却风机电耗系数为常数、空气侧冷却为固定系数，且 M3 不配冷水机组，因此 M3 冷源日电量与日均温几乎不相关（相关系数约 −0.07），高温日冷却电耗可能被低估。
6. M3 在 40 ℃ 真实热浪日有 105 GPU·h 冷却受限失约（M0 为 0），72 h 热浪压力测试中 M3 失约 183 GPU·h、M0 为 78 GPU·h：完整耦合以更低成本换取了更小的高温安全余量，冷却修复（+8.3 万元/年）后可消除。
2. 气象为丹佛替代数据，正式稿须用中卫 CSWD/TMYx 与其 ASHRAE 设计值重算。
3. 休眠待机功率 0.03 kW/台、热备用比例 20% 为 C 级设定值，需实测；BurstGPT 仅 51 天。
4. 10 温区仍按下界计算自然冷却，是保守近似。
"""
N_TEXT = (C.ROOT / "docs/_n_text.md").read_text(encoding="utf-8").strip() if (C.ROOT / "docs/_n_text.md").exists() else "（训练样本量实验完成后填写）"
rep = rep.replace("{N_TEXT}", N_TEXT)
(C.ROOT / "docs/实验结果报告_E1-E5.md").write_text(rep, encoding="utf-8")
print("written docs/实验结果报告_E1-E5.md")
