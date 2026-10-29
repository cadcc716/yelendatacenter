### 3.2 公式—代码逐条映射

| 论文式 | 含义 | 代码位置（`code/model.py::build`） | 实现要点 |
|---|---|---|---|
| (14) | 容量向量与上界, $N\in\mathbb Z_+$ | `V["N"]=M.var(1,0,X["N"],integer=True)`；`V["PV"..."MC"]` | 评估时以 `x_fix` 固定为常数 |
| (15) | 年化投资 $C^{plan}$ | `a = {k: 投资×CRF(i,L)+固定运维}`，作为容量变量目标系数 | 各设备寿命分别计算 CRF |
| (1) | GPU 资源、处理能力、IT 功率 | 注释 `式(1)` 三组 `M.con` | $p_{rk}$=GPU模式功率+主机分摊，与 $p^{sl}$ 同一计量边界 |
| (2) | 实时任务到达即完成；积压递推 | `U["R"]=A^R`；`Bq[t+1]-Bq[t]+U_B=A_B` | 评估模式加缺额松弛 `sR` |
| (3) | 最大延期 $D$ 与日末清零 | `M.row(Σ_{τ≤t}U_B ≥ Σ_{τ≤t-D}A_B)`，`Bq[0]=Bq[T]=0` | 单行求和约束 |
| (4) | 液冷环路热动态 | `C_w(Tw[t+1]-Tw[t]) = Δt(βP^{IT}-Σ_j(q^{FC}+q^{MC}))` | — |
| (5) | 温区选择、温度上下界、初末条件 | `u`为二进制；`Σ_j u=1`；`ΣT̲u ≤ Tw ≤ ΣT̄u`；`Tw[0]=Tw[T]=T_0` | 解耦基准仅保留单一固定温区 |
| (6) | 模式安全温区集合 | `safe=[j: T̄_j+R^{th}h_k ≤ T^{chip,max}]` | 预处理生成 |
| (7) | 两端点温区门控 | `m ≤ N^{max}Σ_{j∈safe}u[j,t]` 与 `u[j,t+1]` | 全部温区安全时自动略去冗余约束 |
| (8) | 自然冷却容量与门控 | `q^{FC}≤φK^{FC}`；`q^{FC}≤φK^{FC,max}u` | $\underline\phi$ 取温区下界，`thermal_tables()` |
| (9) | 机械制冷容量与门控 | `q^{MC}≤K^{MC}`；`q^{MC}≤K^{MC,max}u` | $a^{MC}=1/COP(T̲_j,T_{amb})$，Carnot 比例按 ASHRAE 90.1 标定 |
| (10) | 光伏与电网上限 | `PPV ≤ K^{PV} g^{PV}`；`Pg∈[0,P^{grid,max}]` | 允许弃光、不售电 |
| (11) | 储能电量状态 | `S[t+1]=(1-δ_B)S[t]+Δt(η_cP^{ch}-P^{dis}/η_d)` | — |
| (12) | SOC 边界、倍率、互斥、初末 SOC | `σ_min E ≤ S ≤ σ_max E`；`P ≤ P^B`；`b`；`P^B≤c^{max}E^B` | $b\in\{0,1\}$ 按论文实现（`RELAX_B=False`）；校验表“同时充放电h”=0 |
| (13) | 电功率平衡 | `Pg+PPV+Pdis-Pch = κ_aux P^{IT}+Σ(a^{FC}q^{FC}+a^{MC}q^{MC})+P_0` | $\kappa_{aux}=1+\delta_e+a^{circ}\beta+a^{air}(1-\beta)$ |
| (16) | 情景日运行费用 | `M.row(Cop = Δt Σ c_t Pg + c^{var,B}(Pch+Pdis))` | — |
| (17) | 有限支持矩模糊集 | `data_prep.build()`：特征 $F$；$\hat\mu$=训练日经验均值；bootstrap 估计 $\delta^0$、$v^{0\pm}$ 并经验证集校准 $\kappa_\delta,\kappa_v$；`nonempty_tau()` LP 检验非空；`worst_case_p()` | 若 $\mathcal P_d=\emptyset$ 则按最小放大 τ 扩大并报告 |
| (18) | 两阶段 DRO 目标 | `prob=="DRO"`：内层 $\max_p$ 的 LP 强对偶 → $\lambda_d+\alpha^\top(\hat\mu+\delta)-\beta^\top(\hat\mu-\delta)+\gamma^{+\top}v^++\gamma^{-\top}v^-$，并对每个 $s$ 加 $\lambda_d+(\alpha-\beta)^\top F_s+\gamma^{+\top}[\rho]_+^2+\gamma^{-\top}[-\rho]_+^2\ge C^{op}_{ds}$ | 单一 MILP 精确求解；SP/RO 为同一框架的特例；`solve_cutting_plane()` 为论文第 5 节迭代算法，用于交叉验证 |
