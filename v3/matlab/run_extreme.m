function run_extreme(site, varargin)
% RUN_EXTREME  极端日试验：在 12 个典型日之外加入 3–4 个极端日（p0 = 1/365，所在簇概率相应扣减），
% 比较 SP、DRO、RO 的规划结果与样本外表现（缺额按成因拆分），检验分布鲁棒是否起作用。
%   run_extreme('zhongwei')          使用 data/zhongwei_ext_data.mat（极端日由 Python 预处理选出）
%   结果：results/zhongwei_ext_{SP_S6,DRO_S5,RO_S7}.mat、样本外 csv，以及对照表 results/zhongwei_极端日对照.csv
global CFG
ext = [site '_ext'];
E = load(fullfile(CFG.data, [ext '_data.mat']));
log_line(sprintf('极端日试验：%s，K = %d（典型日 12 + 极端日 %d），极端日：%s', ext, numel(E.med), numel(E.ext_days), E.ext_names));
run_cases(ext, {'S6', 'S5', 'S7'}, 'ccg', varargin{:});
tags = {'SP_S6', 'DRO_S5', 'RO_S7'};
rows = {};
for i = 1:numel(tags)
    for v = {site, ext}
        sv = v{1}; f = fullfile(CFG.res, sprintf('%s_%s.mat', sv, tags{i}));
        if ~exist(f, 'file'), continue; end
        if ~exist(fullfile(CFG.res, sprintf('%s_oos_%s.mat', sv, tags{i})), 'file'), oos_test(sv, tags{i}, site); end
        S = load(f); r = S.r; O = load(fullfile(CFG.res, sprintf('%s_oos_%s.mat', sv, tags{i}))); M = O.M;
        rows(end + 1, :) = {sv, tags{i}, r.x.N, r.x.MC / 1e3, r.x.FC / 1e3, r.x.TES / 1e3, r.x.HP / 1e3, r.obj / 1e4, ...
            (r.inv + 365 * mean(M(:, 2))) / 1e4, 100 * mean(M(:, 3) > 1), sum(M(:, 5)), sum(M(:, 6)), sum(M(:, 7)), ...
            100 * (1 - sum(M(:, 3)) / sum(M(:, 4))), max(M(:, 2)) / 1e4}; %#ok<AGROW>
    end
end
cols = {'数据集', '方法', 'N', 'K_MC/MW', 'K_FC/MW', 'E_TES/MWh', 'K_HP/MW', '规划能源侧成本/万元', '样本外期望能源侧成本/万元', ...
        '缺额日概率/%', '缺额时长/h', 'GPU受限缺额/GPU·h', '冷却受限缺额/GPU·h', '实时满足率/%', '最高日运行成本/万元'};
write_csv(fullfile(CFG.res, [site '_极端日对照.csv']), cols, rows);
log_line(['极端日对照表已写入 ' fullfile(CFG.res, [site '_极端日对照.csv'])]);
end
