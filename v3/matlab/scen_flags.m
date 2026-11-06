function fl = scen_flags(name)
% SCEN_FLAGS  场景开关（执行方案 5.2 节；可行域嵌套 S1⊂S2，S3⊂S4⊂S5）
%   dvfs    false → GPU 固定第1档
%   thermal true  → 温区门控 + 温区漏电 + 气象相关自然冷却/冷机COP + 液冷热惯性 C_w
%           false → 45 ℃ 温水设计：C_w=0 瞬时平衡，自然冷却按 45–50 ℃ 温区，冷机COP设计日值，漏电参考值，无门控
%   tes     false → 不配蓄冷；heat false → 不配热泵
tab = struct('S1', [0 0 0 0], 'S2', [1 0 0 0], 'S3', [1 1 0 0], 'S4', [1 1 1 0], 'S5', [1 1 1 1]);
v = tab.(name);
fl = struct('dvfs', v(1) == 1, 'thermal', v(2) == 1, 'tes', v(3) == 1, 'heat', v(4) == 1);
end
