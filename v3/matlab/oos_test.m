function T = oos_test(site, tag, datasite)
% OOS_TEST  表G 样本外测试：固定规划容量，对全年 365 个历史日逐日求解运行问题
%   oos_test('zhongwei', 'DRO_S5')                     容量取自 results/zhongwei_DRO_S5.mat
%   oos_test('zhongwei_ext', 'DRO_S5', 'zhongwei')     极端日试验：容量取自 zhongwei_ext 的结果，365 日数据相同
%   每日记录：运行成本、实时缺额（按成因拆分：GPU 台数受限 / 冷却与热安全受限）、缺额时长、超期量、购电、碳排放
%   实时缺额成因：该时段工作 GPU 已达装机数（≥ N−1）计为 GPU 受限，否则为冷却与热安全受限
%   结果写入 results/<site>_oos_<tag>.csv 与 .mat
global CFG
if nargin < 3, datasite = site; end
S = load(fullfile(CFG.res, sprintf('%s_%s.mat', site, tag))); r = S.r;
D = load(fullfile(CFG.data, [datasite '_data.mat']));
P = params_v3(); if isfield(r, 'P'), P = r.P; end
scen = 'S5'; if any(strcmp(tag(end-1:end), {'S1', 'S2', 'S3', 'S4'})), scen = tag(end-1:end); end
n = size(D.AR, 1); M = zeros(n, 10);
for i = 1:n
    [Qv, R] = solve_day_fixed(P, D, r.x, i, scen, struct(), 1e-3, 300);
    act = sum(R.mR, 1) + sum(R.mB, 1); gl = act >= r.x.N - 1;
    co2 = sum(P.E_GRID * R.buy - P.E_HEAT * R.HHP);
    M(i, :) = [i, Qv, sum(R.LR), sum(D.AR(i, :)), sum(R.LR > 1), sum(R.LR(gl)), sum(R.LR(~gl)), sum(R.O), sum(R.buy), co2];
    if mod(i, 30) == 0, log_line(sprintf('[样本外 %s %s] %d/%d', site, tag, i, n)); end
end
T = M;
cols = {'day', 'Q', 'LR', 'AR', 'LR_h', 'LR_gpu', 'LR_cool', 'O', 'buy', 'CO2'};
write_csv(fullfile(CFG.res, sprintf('%s_oos_%s.csv', site, tag)), cols, M);
save(fullfile(CFG.res, sprintf('%s_oos_%s.mat', site, tag)), 'M', 'cols', '-v7');
inv = r.inv; lr = M(:, 3);
log_line(sprintf('[样本外 %s %s] 期望能源侧成本 %.2f 万元，最高日 %.2f 万元，缺额日概率 %.1f%%，缺额 %d h（GPU受限 %.0f / 冷却受限 %.0f GPU·h），实时满足率 %.3f%%', ...
    site, tag, (inv + 365 * mean(M(:, 2))) / 1e4, max(M(:, 2)) / 1e4, 100 * mean(lr > 1), sum(M(:, 5)), sum(M(:, 6)), sum(M(:, 7)), 100 * (1 - sum(lr) / sum(M(:, 4)))));
end
