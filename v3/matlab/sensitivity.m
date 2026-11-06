function sensitivity(site, group, vals)
% SENSITIVITY  灵敏度分析（执行方案 5.6 节），S5 两步法 + 邻域搜索
%   sensitivity('zhongwei', 'ALPHA1', [0.2 0.4 0.7 0.9 0.99])     C12、图13
%   sensitivity('zhongwei', 'M', [90 180 730])                    图13（只改变模糊集半径，典型日不重新抽样）
%   sensitivity('zhongwei', 'C_HEAT', [0.10 0.15 0.25 0.30])      C14、图16
%   sensitivity('zhongwei', 'T_CHIP_MAX', [80 83 87 90])          图15
%   sensitivity('zhongwei', 'CO2_PRICE', [50 100 150 200])        图16
%   结果：results/<site>_sens_<group>_<值>.mat 与汇总 results/<site>_sens_<group>.csv
global CFG
D = load(fullfile(CFG.data, [site '_data.mat'])); rows = {};
for v = vals
    f = fullfile(CFG.res, sprintf('%s_sens_%s_%g.mat', site, group, v));
    if ~exist(f, 'file')
        P = params_v3(group, v); P.K = numel(D.med); [a1, ai] = theta_radius(P, numel(D.med));
        log_line(sprintf('==== 灵敏度 %s = %g ====', group, v));
        r = ccg_two_step(P, D, 'S5', [a1 ai], 'ccg', [], struct('ls_max', 30)); r.P = P; %#ok<NASGU>
        save(f, 'r', '-v7');
    end
    S = load(f); r = S.r; x = r.x;
    rows(end + 1, :) = {v, x.N, x.PV / 1e3, x.EB / 1e3, x.FC / 1e3, x.MC / 1e3, x.TES / 1e3, x.HP / 1e3, r.inv / 1e4, (r.obj - r.inv) / 1e4, r.obj / 1e4, r.gap_final}; %#ok<AGROW>
end
write_csv(fullfile(CFG.res, sprintf('%s_sens_%s.csv', site, group)), ...
    {group, 'N', 'PV/MW', 'E_B/MWh', 'K_FC/MW', 'K_MC/MW', 'E_TES/MWh', 'K_HP/MW', '非GPU投资/万元', '最恶劣期望运行成本/万元', '能源侧年化成本/万元', 'gap'}, rows);
end
