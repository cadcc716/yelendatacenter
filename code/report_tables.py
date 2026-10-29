"""把结果 CSV 渲染为 Markdown 表格 (docs/_tables.md), 保证报告数字与程序输出一致。"""
import numpy as np, pandas as pd
import config as C

S = pd.read_csv(C.RES / "summary.csv").set_index("模型")
K = pd.read_csv(C.RES / "checks.csv")
P = pd.read_csv(C.ROOT / "docs/参数与数据来源表.csv")


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                f = fmt.get(c, "{:,.2f}")
                cells.append("—" if pd.isna(v) else f.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


parts = []
parts.append("## T1 参数与数据来源表\n\n" + md(P))
cap = S[["GPU台数", "光伏kWp", "储能kW", "储能kWh", "自然冷却kWth", "机械制冷kWth", "间隙", "求解时间s"]].reset_index()
cap.columns = ["模型", "GPU/台", "光伏/kWp", "储能/kW", "储能/kWh", "自然冷却/kWth", "机械制冷/kWth", "绝对间隙/元", "求解时间/s"]
parts.append("## T2 容量配置\n\n" + md(cap, {"GPU/台": "{:.0f}", "绝对间隙/元": "{:,.0f}", "求解时间/s": "{:.0f}",
                                         **{c: "{:,.0f}" for c in cap.columns[2:7]}}))
S2 = S.copy(); S2["冷源年化"] = S2["年化_FC"] + S2["年化_MC"]
cost = S2[["年化_N", "年化_PV", "冷源年化", "期望电费_经验", "期望电费_最坏", "能源相关费用_经验", "总费用_经验", "总费用_最坏"]] / 1e4
cost = cost.reset_index(); cost.columns = ["模型", "GPU年化", "光伏年化", "冷源年化", "电费(p̂)", "电费(p*)", "能源相关(p̂)",
                                           "总费用(p̂)", "总费用(p*)"]
parts.append("## T3 年化费用分解 (万元/年)\n\n" + md(cost, {c: "{:,.1f}" for c in cost.columns[1:]}))
op = S[["IT电量MWh", "总电量MWh", "购电量MWh", "PUE", "光伏利用率", "自然冷却散热占比", "冷却电量MWh", "机械制冷电量MWh"]].reset_index()
op.columns = ["模型", "IT电量/MWh", "总电量/MWh", "购电/MWh", "PUE", "光伏利用率", "自然冷却散热占比", "冷源电量/MWh", "机械制冷电量/MWh"]
parts.append("## T4 年度运行指标 (经验概率加权)\n\n" + md(op, {"PUE": "{:.4f}", "光伏利用率": "{:.3f}", "自然冷却散热占比": "{:.3f}",
                                                       **{c: "{:,.1f}" for c in op.columns if "MWh" in c}}))
mc = [c for c in S.columns if c.startswith("模式占比_")]
mo = S[mc].fillna(0).reset_index(); mo.columns = ["模型"] + [c.replace("模式占比_", "").replace("|", "-") for c in mc]
parts.append("## T5 GPU 模式工作时长占比\n\n" + md(mo, {c: "{:.3f}" for c in mo.columns[1:]}))
ck = K.drop(columns=["d", "s"]).groupby("模型").agg(["max"]).droplevel(1, axis=1)
ck["液冷温度min"] = K.groupby("模型")["液冷温度min"].min()
ck = ck.reindex(S.index).reset_index()
parts.append("## T6 物理一致性校验 (所有情景日最大值)\n\n" + md(ck, {c: "{:.4g}" for c in ck.columns[1:]}))
oo = S[["样本外年运行费用", "偏移θ=0.5年运行费用", "偏移θ=1.0年运行费用", "日费用CVaR90", "缺额日比例", "实时缺额GPUh", "延期缺额GPUh"]].copy()
oo.iloc[:, :3] /= 1e4; oo = oo.reset_index()
oo.columns = ["模型", "样本外运行费/万元", "偏移θ=0.5/万元", "偏移θ=1/万元", "日费用CVaR90/元", "缺额日比例", "实时缺额/GPU·h", "延期缺额/GPU·h"]
parts.append("## T7 样本外评估 (123 个测试联合日)\n\n" + md(oo, {"缺额日比例": "{:.3f}", "日费用CVaR90/元": "{:,.0f}",
                                                       "实时缺额/GPU·h": "{:,.0f}", "延期缺额/GPU·h": "{:,.0f}"}))
rel = S[[c for c in S.columns if c.endswith("_较基准%")]].reset_index()
rel.columns = ["模型"] + [c.replace("_较基准%", "") for c in rel.columns[1:]]
parts.append("## T8 相对 Baseline 变化率 (%)\n\n" + md(rel, {c: "{:+.2f}" for c in rel.columns[1:]}))
try:
    SE = pd.read_csv(C.RES / "sensitivity.csv")
    se = SE[["参数", "取值", "模型", "GPU台数", "光伏kWp", "储能kWh", "自然冷却kWth", "机械制冷kWth", "能源相关费用_经验", "总费用_经验",
             "总费用_最坏", "PUE", "同时充放电h", "间隙"]].copy()
    for c in ["能源相关费用_经验", "总费用_经验", "总费用_最坏"]: se[c] /= 1e4
    parts.append("## T9 敏感性分析 (费用: 万元/年)\n\n" + md(se, {"取值": "{:g}", "PUE": "{:.4f}", "自然冷却散热占比": "{:.3f}",
                                                         "间隙": "{:,.0f}", "光伏kWp": "{:,.0f}", "自然冷却kWth": "{:,.0f}",
                                                         "机械制冷kWth": "{:,.0f}", "GPU台数": "{:.0f}", "储能kWh": "{:,.0f}",
                                                         "同时充放电h": "{:.0f}"}))
except FileNotFoundError:
    pass
PR = pd.read_csv(C.RES / "probabilities.csv")
pr = PR[PR.模型.isin(["M4-DRO"])].pivot_table(index="类别", columns="情景", values=["经验概率", "最坏概率"]).round(3)
parts.append("## T10 M4-DRO 经验概率与最坏分布\n\n" + pr.to_markdown() if hasattr(pr, "to_markdown") else "")
try:
    CA = pd.read_csv(C.RES / "calibration.csv")[["类别", "kappa_delta", "kappa_v", "非空放大tau", "训练日", "验证日"]]
    parts.append("## T11 模糊集验证集校准 (50 次随机划分中位数)\n\n" + md(CA, {"kappa_delta": "{:.3f}", "kappa_v": "{:.3f}",
                                                                      "非空放大tau": "{:.3f}"}))
except FileNotFoundError:
    pass
try:
    VE = pd.read_csv(C.RES / "verify.csv")[["检验", "设置", "目标值", "绝对间隙", "N", "FC", "MC", "EB", "同时充放电h", "时间s"]]
    parts.append("## T12 数值等价性验证 (缩减算例)\n\n" + md(VE, {"目标值": "{:,.0f}", "绝对间隙": "{:,.0f}", "N": "{:.0f}",
                                                       "FC": "{:,.0f}", "MC": "{:,.0f}", "EB": "{:,.0f}", "同时充放电h": "{:.0f}",
                                                       "时间s": "{:.0f}"}))
except FileNotFoundError:
    pass
(C.ROOT / "docs/_tables.md").write_text("\n\n".join(parts), encoding="utf-8")
print("written docs/_tables.md")
