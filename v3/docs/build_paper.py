"""
组装论文全文（markdown）：题目、摘要、引言、第1–4章（模型稿）、第5章（v4 结果）、第6章、参考文献、附录参数表。
参数表由 code/params.py 直接生成，保证与计算一致；结果图复制到 paper_figs/ 并按正文编号命名。
用法：python build_paper.py   输出 docs/论文全文_v4.md
"""
import os, re, sys, shutil
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(D)
sys.path.insert(0, f"{ROOT}/code")
import params as PR

rd = lambda f: open(f"{D}/{f}", encoding="utf-8").read()

# ---------------- 各部分
abstract = rd("v4_摘要.md") if os.path.exists(f"{D}/v4_摘要.md") else "## 摘要\n\n【待补】\n"
intro = rd("v4_引言.md"); i = intro.index("## 参考文献"); refs = intro[i:].replace("## 参考文献（引言部分）", "## 参考文献").replace("## 参考文献", "## 参考文献"); intro = intro[:i].rstrip() + "\n"
model = rd("模型稿_第0-4章.md"); model = model[model.index("## 1 系统结构与规划框架"):model.index("## 参考文献")].rstrip() + "\n"
model = model.replace("GPU年化投资约占总成本的94%", "GPU年化投资约占总成本的95%")
ch5 = rd("v4_第5章_结果.md")
ch6 = rd("v4_第6章_结论.md") if os.path.exists(f"{D}/v4_第6章_结论.md") else "## 6 结论\n\n【待补】\n"

# ---------------- 表号：字母表 → 数字表（表1–4 为附录参数表，正文结果表从表5 起）
for a, b in (("表E", "表5"), ("表F", "表6"), ("表G", "表7"), ("表A–表D", "附录表A1–表A4")):
    ch5 = ch5.replace(a, b)

# ---------------- 结果图：文件 → 正文编号
FIG = [("图4", "图5_能源侧成本逐级分解", "S1到S5能源侧年化成本的逐级分解"),
       ("图5", "图6_冷热源容量对比", "S1–S5冷热源容量"),
       ("图6", "图7_关键指标相对Baseline变化", "S2–S5关键指标相对S1的变化率"),
       ("图7", "图8_夏季典型日GPU档位", "夏季典型日（6月15日）GPU档位与液冷温度"),
       ("图8", "图9_夏季典型日液冷温度与冷源", "夏季典型日液冷温度与冷源出力"),
       ("图9", "图10_可延期任务时移", "夏季典型日可延期任务处理量与到达量之差"),
       ("图10", "图11_采暖季典型日余热回收", "采暖季典型日（11月22日）液冷温度与余热回收"),
       ("图11", "图13_SP_DRO_RO样本外对比", "含极端日集合的SP、DRO、RO样本外测试"),
       ("图12", "图14_CCG收敛", "含极端日集合时DRO第二步C&CG收敛过程"),
       ("图13", "图15_置信水平与样本量", "置信水平α1与样本量M对年化总成本的影响"),
       ("图14", "图16_算力冗余率", "算力冗余率的影响"),
       ("图15", "图16b_热惯性与蓄冷替代", "液冷环路等效热容对蓄冷与机械制冷容量的影响"),
       ("图16", "图17_芯片温度上限", "芯片温度上限的影响"),
       ("图17", "图18_热价", "热价对余热热泵容量与余热回收量的影响"),
       ("图18", "图19_气候区对比", "三地气候区规划结果对比")]
PF = f"{ROOT}/paper_figs"; os.makedirs(PF, exist_ok=True)
for num, src, cap in FIG:
    s = f"{ROOT}/figures_v4/{src}.png"
    if os.path.exists(s): shutil.copy(s, f"{PF}/{num}.png")
# 在每个小节末插入本节首次提到的图
def insert_figs(text):
    out = []; seen = set(); blocks = re.split(r"(?m)^(?=### )", text)
    for b in blocks:
        figs = [f for f in FIG if re.search(rf"{f[0]}(?![0-9])", b) and f[0] not in seen and os.path.exists(f"{PF}/{f[0]}.png")]
        add = "".join(f"\n![{n} {c}](../paper_figs/{n}.png)\n\n> **{n}** {c}\n" for n, _, c in figs)
        seen |= {f[0] for f in figs}; out.append(b.rstrip() + "\n" + add + "\n")
    return "".join(out)
ch5 = insert_figs(ch5)

# ---------------- 附录参数表（由 params.py 生成）
P = PR.params(); a = PR.annual_cost(P); Z = PR.zone_consts(P)
zn = [f"{lo}–{hi}" for lo, hi in P["ZONES"]]
rows = []
for k, (fr, v, pd_, h) in enumerate(P["GEARS"]):
    safe = [zn[j] for j in range(5) if Z["safe"][k, j]]
    rows.append(f"| {k + 1} | {fr:.1f} | {v:.2f} | {pd_:.3f} | {h:.3f} | {safe[0].split('–')[0]}–{safe[-1].split('–')[1]} ℃ | {'实时、可延期' if k in P['K_R'] else '可延期'} |")
tA = ("**附录表A1 GPU频率档位参数**（单卡服务器基础功率0.45 kW，参考温度40 ℃时漏电0.14 kW，漏电温度系数0.02 /K）\n\n"
      "| 档位 | 频率比 | 吞吐比 | 动态功率/kW | 芯片发热/kW | 安全液冷温区 | 可用任务 |\n| --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows))
tB = f"""**附录表A2 热管理参数**

| 参数 | 取值 |
| --- | --- |
| 芯片温度上限、冷却液—热点热阻 | {P['T_CHIP_MAX']:.0f} ℃，{P['R_TH']:.0f} K/kW |
| 液冷温区 | {'、'.join(zn)} ℃ |
| 液冷份额β、等效热容C_w | {P['BETA']}，{P['C_W']:.0f} kWh/K |
| 自然冷却逼近温差/设计温差、风机电耗系数 | {P['DT_APP']:.0f} K / {P['DT_DES']:.0f} K，{P['A_FC']} |
| 冷机COP（25 ℃、35 ℃基准） | {P['COP0']}，c1 = {P['C1']} /K，c2 = {P['C2']} /K，截断[{P['COP_MIN']:.0f}, {P['COP_MAX']:.0f}] |
| 余热热泵 | 供热{P['T_SUP']:.0f} ℃，η_c = {P['ETA_C']}，COP上限{P['COP_HP_MAX']:.0f} |
| 蓄冷罐 | 自损{100 * P['TES_LOSS']:.1f}%/h，效率{P['TES_EFF']}，最短充放时间{P['TES_TAU']:.0f} h |
| 辅助电耗 | δ_e = {P['DELTA_E']}，a_circ = {P['A_CIRC']}，a_air = {P['A_AIR']}，P0 = {P['P0']:.0f} kW |"""
nm = dict(N="GPU（元/卡）", PV="光伏（元/kW）", EB="储能能量（元/kWh）", PB="储能功率（元/kW）", FC="自然冷却（元/kW_th）",
          MC="机械制冷（元/kW_th）", TES="蓄冷罐（元/kWh_th）", HP="余热热泵（元/kW_th）")
tC = "**附录表A3 设备经济参数**（折现率8%）\n\n| 设备 | 单价 | 寿命/年 | 年运维率 | 安装上限 | 年化费用 |\n| --- | --- | --- | --- | --- | --- |\n" + \
     "\n".join(f"| {nm[k]} | {c[0]:g} | {c[1]} | {c[2]:g} | {c[3]:g} | {a[k]:.1f} |" for k, c in P["COST"].items())
tD = f"""**附录表A4 价格、碳交易与惩罚参数**

| 参数 | 取值 |
| --- | --- |
| 分时电价（峰/平/谷） | {P['PRICE_PEAK']}/{P['PRICE_FLAT']}/{P['PRICE_VALLEY']} 元/kWh；峰段8–11时、17–22时，谷段23–7时 |
| 并网容量 | {P['P_GRID_MAX'] / 1e3:.0f} MW，不向电网反送电 |
| 热价 | {P['C_HEAT']} 元/kWh_th |
| 电网排放因子、替代供热排放因子 | {P['E_GRID'] * 1e3:.2f} kg/kWh，{P['E_HEAT'] * 1e3:.2f} kg/kWh_th |
| 碳配额 | {P['QUOTA'] * 1e3:.1f} kg/(GPU·h) |
| 阶梯碳价 | 基准{P['CO2_PRICE']:.0f}元/t，区间长度{P['CO2_LEN']:.0f} t，每级增幅{100 * P['CO2_GROW']:.0f}%，共{P['CO2_V']}级 |
| 实时任务切除惩罚、可延期任务超期惩罚 | {P['C_R']:.0f} 元/(GPU·h)，{P['C_O']:.0f} 元/(GPU·h) |
| 弃光惩罚、储能损耗成本 | {P['C_CUR']} 元/kWh，{P['C_DEG']} 元/kWh |
| 算力冗余率、截止期 | r_N = {P['R_N']}，D = {P['D_DEADLINE']} h |"""
appendix = "## 附录 参数表\n\n" + "\n\n".join((tA, tB, tC, tD)) + "\n"

head = "# 计及GPU动态调频与液冷热惯性的AI算力园区算—电—热协同分布鲁棒容量规划\n\n" \
       "**Distributionally Robust Capacity Planning of AI Computing Parks with Computing–Power–Thermal Coordination Considering GPU Dynamic Frequency Scaling and Liquid-Cooling Thermal Inertia**\n\n"
full = head + abstract.strip() + "\n\n" + intro + "\n" + model + "\n" + ch5.strip() + "\n\n" + ch6.strip() + "\n\n" + refs.strip() + "\n\n" + appendix
open(f"{D}/论文全文_v4.md", "w", encoding="utf-8").write(full)
print("已生成 论文全文_v4.md，字数约", len(re.sub(r"\s", "", full)), "；待补处：", full.count("【待"))
