%% MAIN_ALL  运行顺序（每一节可单独运行；结果都在 results/ 下，日志 results/run_log.txt）
% 先修改下一行的 YALMIP 路径。Gurobi 需已安装并在 MATLAB 中可用（运行 gurobi_setup 一次）。
init_env('yalmip', 'C:\YALMIP-master');            % ← 改成你的 YALMIP 目录；若已在路径中，写 init_env() 即可

%% 0 自检：单日模型（约 1 分钟）
% S5 第 352 天（2023-12-18），给定容量下的日运行成本，Python/HiGHS 结果为 17945.19 元
D = load(fullfile('data', 'zhongwei_data.mat')); P = params_v3();
x = struct('N', 4508, 'PV', 15000, 'PB', 6703, 'EB', 29628, 'FC', 5172, 'MC', 2104, 'TES', 15745, 'HP', 1448);
[Qv, R] = solve_day_fixed(P, D, x, 352, 'S5');
fprintf('自检：S5 第352天日运行成本 %.2f 元（应为 17945.19）\n', Qv);

%% 1 极端日试验（优先，回答分布鲁棒是否起作用）
% 先跑基准典型日集合的 SP/DRO/RO，再跑加入极端日后的 SP/DRO/RO，最后对照
% 第一轮先用较短的主问题时限（1200 s）看方向；结论确定后再用缺省 3600 s 重算
run_cases('zhongwei', {'S6', 'S5', 'S7'}, 'ccg', 'tl2', 1200);   % S5 会尝试用 S4 结果热启动，没有则跳过
run_extreme('zhongwei', 'tl2', 1200);                             % 生成 results/zhongwei_极端日对照.csv

%% 2 主算例 S1–S7（两步法，第二步 C&CG）及单层 MILP 校验
run_cases('zhongwei');                            % 按 S1→S2、S3→S4→S5、S6、S7 顺序
run_cases('zhongwei', {}, 'mono');                % 单层 MILP(40) 校验（不做邻域搜索）
summarize('zhongwei');                            % 表E：results/zhongwei_tableE.csv

%% 3 表F、表G、图7–10 数据
table_f('zhongwei');
for t = {'DRO_S5', 'SP_S6', 'RO_S7'}, oos_test('zhongwei', t{1}); end
export_days('zhongwei');

%% 4 灵敏度与气候区
sensitivity('zhongwei', 'ALPHA1', [0.2 0.4 0.7 0.9 0.99]);
sensitivity('zhongwei', 'C_HEAT', [0.10 0.15 0.25 0.30]);
sensitivity('zhongwei', 'T_CHIP_MAX', [80 83 87 90]);
sensitivity('zhongwei', 'M', [90 180 730]);
for s = {'guian', 'shaoguan'}, run_cases(s{1}, {'S1', 'S2', 'S3', 'S4', 'S5'}); summarize(s{1}); end
