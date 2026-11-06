# MATLAB + YALMIP + Gurobi 版：运行说明

本目录为论文《计及GPU动态调频与液冷热惯性的AI算力园区算-电-热协同分布鲁棒容量规划》的 MATLAB 实现。
它与仓库中的 Python/HiGHS 版（`v3/code/`）逐条对应，已在 Octave + YALMIP + GLPK 上核对过。
单日运行模型（S1、S2、S5）、两步法规划（S1、S2，2 个典型日）的目标值与 Python 版一致到 0.01 元；
12 个典型日的 S1、S2 两步法 C&CG 与 Python 版相差 0.01%（在求解间隙内），表F 各项一致；
oos_test、summarize、export_days 已跑通。缺额“时长”在惩罚线性时不唯一，论文中以缺额量（GPU·h）为准。

## 1 环境

| 软件 | 要求 |
| --- | --- |
| MATLAB | R2020a 及以上（.m 文件为 UTF-8 编码，含中文注释） |
| YALMIP | 最新版（https://github.com/yalmip/YALMIP ，解压后把目录加入路径） |
| Gurobi | 10.0 及以上，学术许可证；在 MATLAB 中运行一次 `gurobi_setup` |
| Parallel Computing Toolbox | 可选；`init_env('parallel', true)` 时子问题用 parfor |

## 2 目录

```
matlab/
  main_all.m            运行顺序（分节，可逐节运行）
  init_env.m            路径与求解器设置
  params_v3.m           表A–表D 参数（与 Python 版 params.py 一致，含调参记录）
  model_consts.m        温区常数、年化单价、分时电价、自然冷却系数、冷机COP
  build_capacity.m      容量变量与年化投资 式(28)(33)
  build_day.m           单日运行约束 式(1)–(29)
  solve_day_fixed.m     子问题(36)：固定容量求单日运行
  worst_p.m             子问题(37)：最恶劣分布
  solve_dro.m           C&CG（主问题(35)）或等价单层 MILP(40)
  ccg_two_step.m        两步法 + 嵌套热启动 + 邻域搜索（第 4.6 节）
  local_search.m        容量邻域搜索
  run_cases.m           S1–S7 批量求解
  run_extreme.m         极端日试验（SP/DRO/RO 对照）
  oos_test.m            表G 样本外测试（365 日，缺额按成因拆分）
  table_f.m             表F（S1、S2 容量代入完整模型）
  export_days.m         图7–10 逐时数据
  summarize.m           表E
  sensitivity.m         灵敏度
  data/                 中卫、贵安、韶关的 365 日数据与典型日；*_ext 为加入极端日后的场景集合
  results/              运行后生成：各场景 .mat、csv、run_log.txt
```

## 3 运行

1. 打开 `main_all.m`，把第 3 行的 YALMIP 路径改成你的目录。
2. 运行第 0 节“自检”：屏幕应输出 `S5第352天日运行成本 17945.19 元`。数值不一致时先停下，把 `results/run_log.txt` 发给我。
3. 按顺序运行第 1–4 节。各节都会把结果写到 `results/`，可以分几次运行；已有结果的场景会被后续步骤直接读取。

| 节 | 内容 | 预计用时（8 核、Gurobi） |
| --- | --- | --- |
| 1 | 极端日试验：基准集合与加入极端日集合各跑 SP/DRO/RO，加 365 日样本外测试 | 2–4 h |
| 2 | S1–S7 主算例（C&CG）及单层 MILP 校验，表E | 3–6 h |
| 3 | 表F、样本外测试（S5–S7）、图7–10 数据 | 1–2 h |
| 4 | 灵敏度（α1、热价、芯片温限、M）与贵安、韶关 | 6–10 h |

用时为估计值，以实际为准。单个场景的主问题时限在 `ccg_two_step.m` 的缺省参数里（第一步 900 s，第二步 3600 s），可以在调用时修改，例如
`run_cases('zhongwei', {'S5'}, 'ccg', 'tl2', 1800)`。

## 4 求解设置

- 两步法：第一步以总成本为目标求 N*；第二步固定 N*，只以能源侧年化成本为目标，MIPGap 与 C&CG 收敛判据 ε 均为 1e-3（相对能源侧成本）。
- 嵌套热启动：S2←S1、S4←S3、S5←S4，上一场景最优解（新增设备取 0）与第一步容量取优，作初始解、Gurobi Cutoff 与上界初值。
- Gurobi：`MIPFocus = 3`、`Threads = 0`（全部核），可在 `init_env('MIPFocus', 1)` 中改。
- 储能、蓄冷充放互斥变量松弛为连续变量，求解后逐时段校验；出现同时充放时只在该时段恢复 0-1 变量。

## 5 运行完请发回

把整个 `results/` 文件夹压缩发给我（.mat 较大时，至少发下列文件）：

1. `run_log.txt`（全部日志）
2. `zhongwei_极端日对照.csv`（第 1 节）
3. `zhongwei_tableE.csv`、`zhongwei_tableF.csv`、`zhongwei_oos_*.csv`
4. `fig_data/` 下的逐时 csv
5. 灵敏度汇总 `zhongwei_sens_*.csv`

出错时请发：MATLAB 报错全文、`run_log.txt`、MATLAB 与 Gurobi 版本号。

## 6 与 Python 版结果的对照

`reference_python/` 下是 Python/HiGHS 版在同一数据上的结果（HiGHS 的 gap 为 0.5–1.3%），供核对：

- `zhongwei_tableE.csv`、`zhongwei_tableE_mono.csv`：S1–S7 容量与成本
- `zhongwei_tableF.csv`：表F
- `zhongwei_oos_*.csv`：样本外测试
- `extreme_preview.txt`：极端日试验第一步的预览（DRO 的 N* 为 4580，SP 为 4508）

Gurobi 的 gap 更小，结果与 HiGHS 版相差应在 HiGHS gap 范围内；若出现明显差异（例如容量方向相反），请一并发回。
