function f = newfig(h_cm)
% 单栏宽 8.5 cm，高度 h_cm（cm）
f = figure('Units', 'centimeters', 'Position', [2 2 8.5 h_cm], 'Color', 'w', 'PaperPositionMode', 'auto');
end
