function summarize(site)
% SUMMARIZE  表E：S1–S7 容量、成本分项、运行指标、求解信息；C&CG 与单层 MILP 对照。写 results/<site>_tableE.csv
% 口径：第二步目标即能源侧年化成本 = 非 GPU 年化投资 + 365·Σ p*_s Q_s；总成本 = 能源侧 + GPU 年化投资（N*）
%       成本分项按同一 p* 加权；运行指标按 p0 加权；PUE 不含热泵用电；ERF = 回收热量 / 数据中心总用电
global CFG
D = load(fullfile(CFG.data, [site '_data.mat']));
tags = {'S1', 'DRO_S1'; 'S2', 'DRO_S2'; 'S3', 'DRO_S3'; 'S4', 'DRO_S4'; 'S5', 'DRO_S5'; 'S6', 'SP_S6'; 'S7', 'RO_S7'};
cols = {'场景', 'N', 'PV/MW', 'P_B/MW', 'E_B/MWh', 'K_FC/MW', 'K_MC/MW', 'E_TES/MWh', 'K_HP/MW', 'GPU投资/万元', '其他投资/万元', ...
        '购电/万元', '售热/万元', '碳交易/万元', '惩罚/万元', '能源侧年化成本/万元', '总成本/万元', '第一步N*', '第一步总成本/万元', ...
        '名义PUE', 'ERF/%', '实时满足率/%', '超期量/GPU·h', '碳排放/t', '余热回收/MWh', '弃光率/%', '自然冷却利用小时/h', ...
        '峰值IT功率/MW', 'C&CG迭代次数', '最终gap', '求解时间/s', '邻域搜索前/万元', '单层MILP/万元', '两法相对差/%'};
rows = {};
for i = 1:size(tags, 1)
    f = fullfile(CFG.res, sprintf('%s_%s.mat', site, tags{i, 2}));
    if ~exist(f, 'file'), continue; end
    S = load(f); r = S.r; P = r.P; x = r.x; p0 = r.p0(:); pw = r.p(:); K = numel(r.R);
    it = zeros(K, 13);
    for s = 1:K
        R = r.R{s}; lam = P.CO2_PRICE;
        co2c = lam * (1 + (0:P.CO2_V - 1) * P.CO2_GROW) * R.eps - lam * R.eps_s;
        pen = P.C_R * sum(R.LR) + P.C_O * sum(R.O) + P.C_CUR * sum(R.cur) + P.C_DEG * sum(R.Pch + R.Pdis);
        Et = sum(R.PIT * (1 + R.consts.kaux) + P.P0 + R.PFC + R.PMC);
        it(s, :) = [R.consts.price * R.buy', -P.C_HEAT * sum(R.HHP), co2c, pen, Et, sum(R.PIT), sum(R.cur), sum(R.PV + R.cur), ...
                    sum(R.LR), sum(D.AR(D.med(s), :)), sum(R.O), sum(P.E_GRID * R.buy - P.E_HEAT * R.HHP), sum(R.HHP)];
        fch(s) = sum(sum(R.qFC, 1) > 1); pit(s) = max(R.PIT); %#ok<AGROW>
    end
    Ew = @(c) 365 * (pw' * it(:, c)); E0 = @(c) 365 * (p0' * it(:, c));
    gpu = r.a_full.N * x.N; mono = NaN; dif = NaN;
    fm = fullfile(CFG.res, sprintf('%s_%s_mono.mat', site, tags{i, 2}));
    if exist(fm, 'file'), Sm = load(fm); mono = Sm.r.obj; dif = 100 * (mono - r.obj_pre_ls) / r.obj_pre_ls; end
    rows(end + 1, :) = {tags{i, 1}, x.N, x.PV / 1e3, x.PB / 1e3, x.EB / 1e3, x.FC / 1e3, x.MC / 1e3, x.TES / 1e3, x.HP / 1e3, gpu / 1e4, r.inv / 1e4, ...
        Ew(1) / 1e4, Ew(2) / 1e4, Ew(3) / 1e4, Ew(4) / 1e4, r.obj / 1e4, (r.obj + gpu) / 1e4, r.step1.N, r.step1.obj_total / 1e4, ...
        E0(5) / E0(6), 100 * E0(13) / E0(5), 100 * (1 - E0(9) / E0(10)), E0(11), E0(12), E0(13) / 1e3, 100 * E0(7) / max(E0(8), 1e-9), ...
        365 * (p0' * fch(:)), max(pit) / 1e3, numel(r.hist), r.gap_final, r.t_total, r.obj_pre_ls / 1e4, mono / 1e4, dif}; %#ok<AGROW>
    clear fch pit
end
write_csv(fullfile(CFG.res, [site '_tableE.csv']), cols, rows);
log_line(['表E 已写入 ' fullfile(CFG.res, [site '_tableE.csv'])]);
for i = 1:size(rows, 1)
    log_line(sprintf('%s: N %d, MC %.2f MW, TES %.1f MWh, HP %.2f MW, 能源侧 %.2f 万元, 碳 %.0f t, PUE %.3f, gap %.2e', ...
        rows{i, 1}, rows{i, 2}, rows{i, 7}, rows{i, 8}, rows{i, 9}, rows{i, 16}, rows{i, 24}, rows{i, 20}, rows{i, 30}));
end
% 单调性检查：嵌套场景的能源侧年化成本须满足 S2 ≤ S1、S4 ≤ S3、S5 ≤ S4（N* 相同时）
pairs = {'S2', 'S1'; 'S4', 'S3'; 'S5', 'S4'};
for k = 1:size(pairs, 1)
    a = find(strcmp(rows(:, 1), pairs{k, 1})); b = find(strcmp(rows(:, 1), pairs{k, 2}));
    if isempty(a) || isempty(b), continue; end
    if rows{a, 2} ~= rows{b, 2}
        log_line(sprintf('单调性检查 %s vs %s：N* 不同（%d / %d），不比较', pairs{k, 1}, pairs{k, 2}, rows{a, 2}, rows{b, 2}));
    elseif rows{a, 16} <= rows{b, 16} * (1 + 1e-6)
        log_line(sprintf('单调性检查 %s ≤ %s：通过（%.2f ≤ %.2f 万元）', pairs{k, 1}, pairs{k, 2}, rows{a, 16}, rows{b, 16}));
    else
        log_line(sprintf('!!!! 单调性检查 %s ≤ %s 未通过（%.2f > %.2f 万元）：用 run_cases(''%s'', {''%s''}, ''ccg'', ''force'', true) 重算 %s', ...
            pairs{k, 1}, pairs{k, 2}, rows{a, 16}, rows{b, 16}, site, pairs{k, 1}, pairs{k, 1}));
    end
end
end
