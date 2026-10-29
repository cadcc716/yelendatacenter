# yelendatacenter

考虑GPU运行调节与液冷热状态耦合的AI数据中心分布鲁棒容量规划模型 —— 数据、代码与实验闭环。

| 路径 | 内容 |
|---|---|
| `doit` | 论文数学模型（旧版式1–18）；修订版模型与实验方案见 `main` 分支《数学模型与完整实验方案_修订版》 |
| `docs/实验结果报告_E1-E5.md` | **修订版实验方案 E1–E5 结果报告**（统一物理复核、交互项、同容量运行、不确定性前沿、压力测试、设计日校核、气候边界、求解效率） |
| `docs/论文实验闭环.md` | **完整实验报告**：数据来源表、公式-代码映射、结果校验、定量比较、机理解释、图件说明、局限 |
| `docs/代码一致性审查.md` | 论文模型与代码逐项一致性审查：差异清单、影响、修改与数值验证 |
| `docs/参数与数据来源表.csv` | 可直接放入论文的参数表（含来源、年份、可信度、使用位置） |
| `code/config.py` | 全部参数（每项带来源与可信度标签）；`params()` 派生参数随覆盖值联动重算 |
| `code/data_prep.py` | 负荷轨迹/气象/光伏/电价 → 代表日情景与式(17)模糊集（经验均值、bootstrap、验证集校准、非空性检验） |
| `code/model.py` | MILP 构建（式1–18 逐条对应）与 HiGHS 求解；`solve_cutting_plane()` 为论文第5节迭代算法 |
| `code/run_all.py` | 7 个模型场景、物理校验、样本外评估、敏感性分析（`job`/`sjob` 单作业 + `collect` 汇总） |
| `code/run_parallel.sh` | 并行驱动：数据处理 → 7 个模型 + 16 个敏感性算例 → 汇总 |
| `code/verify.py` | 数值验证：V1 收紧大M=论文原式、V2 对偶重构=割平面迭代、V3 储能互斥 |
| `code/data_e.py` | 修订版数据集：时间顺序训练/验证/测试划分、代表日+极端日+ASHRAE 设计极端日、U0/U1/U2/U3 模糊集与验证块校准、连续压力片段 |
| `code/exp.py` | 修订版 E1–E5 实验：方案定义、统一物理评价器、修复诊断、同容量运行比较、设计日校核、求解效率；`run_exp.sh`/`run_exp2.sh` 并行驱动 |
| `code/plot_e.py`, `code/report_e.py` | E1–E5 图件（`figures/exp/`，17 张）与报告生成 |
| `code/report_tables.py`, `code/build_report.py` | 结果 CSV → Markdown 表格 → 组装报告 |
| `code/plot_a.py` | 图组A：23 pt Times New Roman 时序对比图 |
| `code/plot_b.py` | 图组B：8.5 cm 单栏 600 dpi 重设计图（Baseline 对比、差值/变化率） |
| `code/diagrams.py` | 论文示意图：F1 系统结构图、F2 研究框架图、F3 求解流程图（PDF/SVG/PNG） |
| `data/raw` | NREL TMY3 气象文件；BurstGPT 原始轨迹（未入库，自动下载） |
| `data/processed` | 逐时聚合轨迹、情景与模糊集 |
| `results/exp/` | E1–E5 全部结果：summary.csv、experiments.xlsx、逐日复核 hf_*.csv、压力测试、设计日校核、同容量运行、求解效率 |
| `results/` | summary/checks/timeseries/oos/sensitivity/calibration/verify CSV 与 results.xlsx；`old_v1/` 为审查前结果 |
| `figures/` | exp（E1–E5 结果图）；diagrams 矢量示意图；style_a / style_b（旧版实验图） |

## 复现

```bash
pip install numpy scipy pandas matplotlib highspy pvlib openpyxl
```

```bash
cd code && NJ=4 ./run_parallel.sh && python verify.py && python plot_a.py && python plot_b.py && python diagrams.py && python report_tables.py && python build_report.py
```

更换为真实站点（如宁夏中卫 CSWD 典型年）：把 EPW 文件放入 `data/raw/`，运行 `python data_prep.py --epw 文件名.epw`。

### 修订版实验 E1–E5

```bash
cd code && python data_e.py && NJ=4 ./run_exp.sh && python exp.py design && python exp.py collect && python plot_e.py && python report_e.py
```
