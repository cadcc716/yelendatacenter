function export_days(site, dates)
% EXPORT_DAYS  图7–图10 数据：固定 S5 规划容量，求解指定日期的运行问题，导出逐时数据与档位分布
%   export_days('zhongwei')                                  6 月 15 日 + 采暖季中 p0 最大的典型日
%   export_days('zhongwei', {'2023-06-15', '2023-12-18'})
global CFG
D = load(fullfile(CFG.data, [site '_data.mat'])); S = load(fullfile(CFG.res, sprintf('%s_DRO_S5.mat', site))); r = S.r;
P = params_v3(); if isfield(r, 'P'), P = r.P; end
if nargin < 2
    med = D.med(:); heat = D.heating(med) > 0; pm = D.p0(:); pm(~heat) = -1; [~, iw] = max(pm);
    dates = {'2023-06-15', D.dates(med(iw), :)};
end
out = fullfile(CFG.res, 'fig_data'); if ~exist(out, 'dir'), mkdir(out); end
for k = 1:numel(dates)
    i = find(all(D.dates == repmat(dates{k}, size(D.dates, 1), 1), 2), 1);
    [Qv, R] = solve_day_fixed(P, D, r.x, i, 'S5', struct(), 1e-4, 1200);
    J = R.consts.J; g = zeros(P.T, 5);
    for a = 1:numel(P.K_R), g(:, P.K_R(a)) = g(:, P.K_R(a)) + sum(R.mR((a - 1) * J + (1:J), :), 1)'; end
    for a = 1:numel(P.K_B), g(:, P.K_B(a)) = g(:, P.K_B(a)) + sum(R.mB((a - 1) * J + (1:J), :), 1)'; end
    write_csv(fullfile(out, sprintf('%s_%s_档位.csv', site, dates{k})), {'时', '档1', '档2', '档3', '档4', '档5'}, [(0:P.T - 1)', g]);
    [~, zone] = max(R.u, [], 1);
    H = [(0:P.T - 1)', D.Tamb(i, :)', R.Tw(1:P.T)', zone', R.PIT', sum(R.qFC, 1)', sum(R.qMC, 1)', sum(R.qHP, 1)', R.qch', R.qdis', R.S(1:P.T)', ...
         R.HHP', D.H(i, :)', R.PHP', R.PMC', R.PFC', (D.gPV(i, :) * r.x.PV)', R.PV', R.cur', R.buy', R.Pch', R.Pdis', R.SB(1:P.T)', R.consts.price', ...
         D.AR(i, :)', R.UR', R.LR', D.AB(i, :)', R.UB', R.B(1:P.T)', R.O'];
    cols = {'时', '气温', '液冷温度', '温区', 'P_IT', '自然冷却', '机械制冷', '热泵取热', '蓄冷充冷', '蓄冷放冷', '蓄冷库存', '热泵供热', '热需求', ...
            '热泵用电', '冷机用电', '干冷器用电', '光伏可用', '光伏实用', '弃光', '购电', '储能充电', '储能放电', '储能SOC', '电价', ...
            '实时到达', '实时处理', '实时切除', '可延期到达', '可延期处理', '可延期积压', '超期量'};
    write_csv(fullfile(out, sprintf('%s_%s_逐时.csv', site, dates{k})), cols, H);
    k1 = g(:, 1) ./ max(sum(g, 2), 1); val = ismember(0:P.T - 1, P.VALLEY_H);
    log_line(sprintf('%s：日运行成本 %.2f 万元；12–16时 k1 占比 %.0f%%，0–6时 %.0f%%；谷时段可延期处理占比 %.0f%%；T_w≥45℃ 时段占比 %.0f%%', ...
        dates{k}, Qv / 1e4, 100 * mean(k1(13:16)), 100 * mean(k1(1:6)), 100 * sum(R.UB(val)) / sum(R.UB), 100 * mean(R.Tw(1:P.T) >= 45 - 1e-6)));
end
end
