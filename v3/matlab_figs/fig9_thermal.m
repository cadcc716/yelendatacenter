% 图9 夏季典型日液冷温度与冷源（S3、S4、S5）
sci_setup; T = rd('图9_夏季典型日液冷温度与冷源'); f = newfig(7.6);
sc = {'S3', 'S4', 'S5'}; ls = {'--', ':', '-'}; col = [0 160 135; 60 84 136; 230 75 53]/255;
subplot(2, 1, 1); hold on; peak = [8 11; 17 22];
for k = 1:2, patch([peak(k,1) peak(k,2) peak(k,2) peak(k,1)], [28 28 56 56], [0.62 0.62 0.62], 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off'); end
for k = 1:3
    m = strcmp(T.('场景'), sc{k}); if ~any(m), continue; end
    plot(T.('时刻')(m) + 0.5, T.('液冷温度')(m), ls{k}, 'Color', col(k, :), 'Marker', 'o', 'DisplayName', sc{k});
end
ylabel(mix('液冷温度/℃')); xlim([0 24]); set(gca, 'XTick', 0:4:24); lg = legend('Location', 'northwest', 'NumColumns', 3); lg.Box = 'off';
title('(a)', 'FontWeight', 'normal', 'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0 1.0]);
subplot(2, 1, 2); hold on;
m5 = strcmp(T.('场景'), 'S5'); bar(T.('时刻')(m5) + 0.5, T.('蓄冷放冷_kW')(m5)/1e3, 0.8, 'FaceColor', [60 84 136]/255, 'FaceAlpha', 0.35, 'EdgeColor', 'none', 'DisplayName', mix('S5蓄冷放冷'));
for k = 1:3
    m = strcmp(T.('场景'), sc{k}); if ~any(m), continue; end
    plot(T.('时刻')(m) + 0.5, T.('机械制冷_kW')(m)/1e3, ls{k}, 'Color', col(k, :), 'Marker', 's', 'DisplayName', mix([sc{k} '机械制冷']));
end
ylabel(mix('冷量/MW')); xlabel(mix('时刻/h')); xlim([0 24]); set(gca, 'XTick', 0:4:24); lg = legend('Location', 'northwest', 'NumColumns', 2); lg.Box = 'off';
title('(b)', 'FontWeight', 'normal', 'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0 1.0]);
savefig600(f, '图9_夏季典型日液冷温度与冷源');
