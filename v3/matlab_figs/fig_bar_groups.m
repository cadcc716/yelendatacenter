% 图6、图7、图19 等分组柱状图的通用绘制示例（图7：关键指标相对 Baseline 变化率）
sci_setup; T = rd('图7_关键指标相对Baseline变化'); f = newfig(5.8);
sc = unique(T.('场景'), 'stable'); ind = unique(T.('指标'), 'stable'); ind = ind(~strcmp(ind, 'PUE'));
Y = zeros(numel(sc), numel(ind));
for i = 1:numel(sc), for j = 1:numel(ind)
    Y(i, j) = T.('相对S1变化率_百分比')(strcmp(T.('场景'), sc{i}) & strcmp(T.('指标'), ind{j}));
end, end
b = bar(Y, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.3); cols = [60 84 136; 0 160 135; 77 187 213; 243 155 127]/255;
for j = 1:numel(b), b(j).FaceColor = cols(j, :); end
yline(0, 'k', 'LineWidth', 0.6); set(gca, 'XTickLabel', cellfun(@(s) mix([s '相对S1']), sc, 'UniformOutput', false));
ylabel(mix('相对Baseline(S1)变化率/%')); lg = legend(cellfun(@mix, ind, 'UniformOutput', false), 'Orientation', 'horizontal', 'Location', 'northoutside');
lg.Box = 'off'; savefig600(f, '图7_关键指标相对Baseline变化');
