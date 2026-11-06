function table_f(site)
% TABLE_F  表F：把 S1、S2 的容量代入 S5 完整模型（门控 + 漏电 + 气象冷却 + 热惯性；不配蓄冷、热泵）逐典型日运行，
% 统计实时缺额、超期量（按 p0 年化）与复核 PUE；并统计 S1/S2 自身名义运行中的热越限时长与最大超温。
global CFG
D = load(fullfile(CFG.data, [site '_data.mat'])); P = params_v3(); Cz = model_consts(P);
h = P.GEARS(:, 4); rows = {};
for tg = {'S1', 'S2'}
    S = load(fullfile(CFG.res, sprintf('%s_DRO_%s.mat', site, tg{1}))); r = S.r;
    x = r.x; x.TES = 0; x.HP = 0; K = numel(D.med); res = zeros(K, 7); hrs = zeros(K, 1); dT = zeros(K, 1);
    for s = 1:K
        [Qv, R] = solve_day_fixed(P, D, x, D.med(s), 'S5');
        Etot = sum(R.PIT * (1 + R.consts.kaux) + P.P0 + R.PFC + R.PMC);
        res(s, :) = [Qv, sum(R.LR), sum(R.LR > 1), sum(D.AR(D.med(s), :)), sum(R.O), Etot, sum(R.PIT)];
        % 名义运行热越限：S1/S2 自身解中所用档位不在当前温区安全集合内
        Rn = r.R{s}; [~, zone] = max(Rn.u, [], 1); nR = numel(P.K_R); nB = numel(P.K_B); J = Cz.J; m = zeros(5, P.T);
        for t = 1:P.T
            for i = 1:nR, m(P.K_R(i), t) = m(P.K_R(i), t) + Rn.mR((i - 1) * J + zone(t), t); end
            for i = 1:nB, m(P.K_B(i), t) = m(P.K_B(i), t) + Rn.mB((i - 1) * J + zone(t), t); end
        end
        Tchip = repmat(Cz.hi(zone)', 5, 1) + P.R_TH * repmat(h, 1, P.T);
        over = (m > 1) & (Tchip > P.T_CHIP_MAX + 1e-9);
        hrs(s) = sum(any(over, 1)); if any(over(:)), dT(s) = max(Tchip(over) - P.T_CHIP_MAX); end
    end
    p0 = D.p0(:); E = @(c) 365 * (p0' * res(:, c));
    [~, iw] = max(res(:, 2) + 1e3 * res(:, 3));
    rows(end + 1, :) = {tg{1}, E(6) / E(7), E(2), 100 * (1 - E(2) / E(4)), E(3), E(5), 365 * (p0' * hrs), max(dT), ...
        D.dates(D.med(iw), :), res(iw, 3), res(iw, 2), hrs(iw)}; %#ok<AGROW>
end
cols = {'方案', '复核PUE', '年实时缺额/GPU·h', '实时满足率/%', '年缺额时长/h', '年超期量/GPU·h', '名义运行热越限时长/h', ...
        '名义运行最大超温/K', '最不利典型日', '最不利日缺额时长/h', '最不利日缺额/GPU·h', '最不利日名义热越限/h'};
write_csv(fullfile(CFG.res, [site '_tableF.csv']), cols, rows);
log_line('表F 已写入');
end
