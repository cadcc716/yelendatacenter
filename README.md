# yelendatacenter

考虑GPU运行调节与液冷热状态耦合的AI数据中心分布鲁棒容量规划模型 —— 数据、代码与实验闭环。

| 路径 | 内容 |
|---|---|
| `doit` | 论文数学模型（式1–18） |
| `docs/论文实验闭环.md` | **完整实验报告**：数据来源表、公式-代码映射、结果校验、定量比较、机理解释、图件说明、局限 |
| `docs/参数与数据来源表.csv` | 可直接放入论文的参数表（含来源、年份、可信度、使用位置） |
| `code/config.py` | 全部参数（每项带来源与可信度标签） |
| `code/data_prep.py` | 负荷轨迹/气象/光伏/电价 → 代表日情景与式(17)模糊集 |
| `code/model.py` | MILP 构建（式1–18 逐条对应）与 HiGHS 求解 |
| `code/run_all.py` | 6+1 个模型场景、物理校验、样本外评估、敏感性分析、结果导出 |
| `code/sens_extra.py` | 敏感性分析并行补跑（可选） |
| `code/report_tables.py`, `code/build_report.py` | 结果 CSV → Markdown 表格 → 组装报告 |
| `code/plot_a.py` | 图组A：23 pt Times New Roman 时序对比图 |
| `code/plot_b.py` | 图组B：8.5 cm 单栏 600 dpi 重设计图（差值/变化率/局部放大） |
| `data/raw` | NREL TMY3 气象文件；BurstGPT 原始轨迹（未入库，自动下载） |
| `data/processed` | 逐时聚合轨迹、情景与模糊集 |
| `results/` | summary/checks/timeseries/oos/sensitivity CSV 与 results.xlsx |
| `figures/` | style_a / style_b PNG |

## 复现

```bash
pip install numpy scipy pandas matplotlib highspy pvlib openpyxl
```

```bash
cd code && python data_prep.py && python run_all.py all && python plot_a.py && python plot_b.py
```

更换为真实站点（如宁夏中卫 CSWD 典型年）：把 EPW 文件放入 `data/raw/`，运行 `python data_prep.py --epw 文件名.epw`。
