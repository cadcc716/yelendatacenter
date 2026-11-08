% 图5 能源侧年化成本逐级分解（相对 S1 的变化量）
sci_setup; T = rd('图5_能源侧成本逐级分解'); f = newfig(5.6);
d = T.('变化量_万元'); n = numel(d) - 1; cum = 0; hold on;
for i = 1:n
    c = [0 160 135]/255; if d(i) > 0, c = [230 75 53]/255; end
    patch([i-0.3 i+0.3 i+0.3 i-0.3], [cum cum cum+d(i) cum+d(i)], c, 'EdgeColor', 'k', 'LineWidth', 0.4);
    text(i, cum + d(i), sprintf('%+.1f\n(%+.1f%%)', d(i), T.('相对S1_百分比')(i)), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'FontSize', 8, 'FontName', 'Times New Roman');
    cum = cum + d(i);
end
patch([n+1-0.3 n+1+0.3 n+1+0.3 n+1-0.3], [0 0 cum cum], [60 84 136]/255, 'EdgeColor', 'k', 'LineWidth', 0.4);
text(n+1, cum, sprintf('%+.1f\n(%+.1f%%)', cum, T.('相对S1_百分比')(end)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 8);
yline(0, 'k', 'LineWidth', 0.6); xlim([0.4 n+1.6]);
set(gca, 'XTick', 1:n+1, 'XTickLabel', cellfun(@mix, [T.('环节')(1:n); {'S5相对S1'}], 'UniformOutput', false));
ylabel(mix('能源侧年化成本变化/万元')); savefig600(f, '图5_能源侧成本逐级分解');
