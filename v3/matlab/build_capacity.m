function [X, F, inv, a] = build_capacity(P, fl, mode, val)
% BUILD_CAPACITY  容量变量与年化投资 式(28)(33)
%   mode = 'free' : 全部容量为决策变量（两步法第一步；N 取连续变量，收敛后取整）
%          'fixN' : N 固定为 val 且不计其年化投资（两步法第二步，目标只含能源侧年化成本）
%          'fix'  : 全部容量固定为结构体 val（子问题(36)、样本外测试、表F）
%   X 的各字段为 sdpvar 或数值；inv 为年化投资表达式（元/年）
C = model_consts(P); a = C.a; X = struct(); F = []; inv = 0;
for i = 1:numel(P.CAPS)
    k = P.CAPS{i}; ub = C.xmax.(k);
    if strcmp(k, 'TES') && ~fl.tes, ub = 0; end
    if strcmp(k, 'HP') && ~fl.heat, ub = 0; end
    switch mode
        case 'fix'
            if isfield(val, k), X.(k) = min(val.(k), ub); else, X.(k) = 0; end   % 场景不含的设备取 0
            inv = inv + a.(k) * X.(k);
        case 'fixN'
            if strcmp(k, 'N')
                X.N = val;                       % 常数，不计入目标
            else
                X.(k) = sdpvar(1, 1); F = [F, 0 <= X.(k) <= ub];
                inv = inv + a.(k) * X.(k);
            end
        otherwise
            X.(k) = sdpvar(1, 1); F = [F, 0 <= X.(k) <= ub];
            inv = inv + a.(k) * X.(k);
    end
end
if ~strcmp(mode, 'fix')
    F = [F, X.PB <= P.C_B_MAX * X.EB];
end
if strcmp(mode, 'fixN'), a.N = 0; end
end
