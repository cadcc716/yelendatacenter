%% MAIN_ALL  一键运行全部计算（v2）。可整体运行，也可逐节运行；全部支持断点续跑：
%  电脑重启或中断后，重新运行即可，已完成且有效的结果自动跳过。日志：results/run_log.txt
%  统一设置：两步法；第二步主问题时限 1200 s、MIPGap 1e-3；邻域搜索最多 30 次评估；Gurobi 12 线程。
%  预计用时（i7-14650HX）：第1节 8–10 h，第2节 4–6 h，第3节 2–3 h，第4节 12–16 h。
init_env('yalmip', 'C:\YALMIP-master');            % ← 改成你的 YALMIP 目录；若已在路径中，写 init_env() 即可

%% 0 自检：单日模型（约 1 分钟）。应输出 17945.19 元，不一致时先停下
D = load(fullfile('data', 'zhongwei_data.mat')); P = params_v3();
x = struct('N', 4508, 'PV', 15000, 'PB', 6703, 'EB', 29628, 'FC', 5172, 'MC', 2104, 'TES', 15745, 'HP', 1448);
[Qv, R] = solve_day_fixed(P, D, x, 352, 'S5');
fprintf('自检：S5 第352天日运行成本 %.2f 元（应为 17945.19）\n', Qv);

%% 1 极端日试验（回答分布鲁棒是否起作用）
% 基准典型日集合与加入极端日集合各求 SP(S6)/DRO(S5)/RO(S7)，再做 365 日样本外测试
% 完成标志：日志出现“极端日对照表已写入”；结果 results/zhongwei_极端日对照.csv
run_extreme('zhongwei');

%% 2 主算例 S1–S7 与单层 MILP 校验
% S1→S2、S3→S4→S5 嵌套热启动；第 1 节算过的 S5 若当时未用 S4 热启动，会自动重算
run_cases('zhongwei');                            % C&CG
run_cases('zhongwei', {}, 'mono');                % 单层 MILP(40) 校验（不做邻域搜索）
summarize('zhongwei');                            % 表E：results/zhongwei_tableE.csv，日志中含单调性检查

%% 3 表F、表G、图7–10 数据
table_f('zhongwei');                              % results/zhongwei_tableF.csv
for t = {'DRO_S5', 'SP_S6', 'RO_S7'}, oos_test('zhongwei', t{1}); end   % 容量更新过的会自动重算
export_days('zhongwei');                          % results/fig_data/
run_extreme('zhongwei');                          % 刷新极端日对照表（只重算已更新的样本外结果）

%% 4 灵敏度与气候区
sensitivity('zhongwei', 'ALPHA1', [0.2 0.4 0.7 0.9 0.99]);     % C12、图13
sensitivity('zhongwei', 'C_HEAT', [0.10 0.15 0.25 0.30]);      % C14、图16
sensitivity('zhongwei', 'T_CHIP_MAX', [80 83 87 90]);          % 图15
sensitivity('zhongwei', 'M', [90 180 730]);                    % 图13
sensitivity('zhongwei', 'CO2_PRICE', [50 150 200]);            % 图16
for s = {'guian', 'shaoguan'}, run_cases(s{1}, {'S1', 'S3', 'S4', 'S5'}); summarize(s{1}); end
climate_compare({'zhongwei', 'guian', 'shaoguan'});            % C13、图14：results/气候区对比.csv
log_line('全部计算完成');
