function T = rd(name)
% 读取 figures_v4/data 下的图数据（UTF-8 带 BOM，中文列名原样保留）
p = fullfile('..', 'figures_v4', 'data', [name '.csv']);
T = readtable(p, 'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
end
