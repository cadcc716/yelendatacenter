function [val, lb, ok, info] = run_opt(F, obj, ops, has_int)
% RUN_OPT  调用 optimize，返回目标值、下界（Gurobi 的 best bound）与是否得到可行解。
%   下界取 Gurobi 返回的 objbound（需 sdpsettings 中 savesolveroutput = 1，见 solver_ops）；
%   若取不到下界：纯 LP 取目标值；含整数变量时取 -Inf 并给出警告，避免 C&CG 误判收敛。
if nargin < 4, has_int = true; end
diag = optimize(F, obj, ops);
val = value(obj); info = diag;
ok = ~isnan(val) && any(diag.problem == [0 3 4 5 9 15]) && isfinite(val);
lb = -Inf; got = false;
try
    r = diag.solveroutput.result;
    if isfield(r, 'objbound') && isfield(r, 'objval') && ~isempty(r.objbound) && isfinite(r.objbound)
        lb = val - (r.objval - r.objbound);         % 消除 YALMIP 常数项偏移
        got = true;
    end
catch
end
if ~got
    if ~has_int || ~strcmp(ops.solver, 'gurobi')
        lb = val;                                    % LP 或测试求解器：目标值即为最优值
    else
        warning('run_opt: 未取到 Gurobi 的 best bound，下界记为 -Inf');
    end
end
lb = min(lb, val);
if ~ok, val = NaN; end
end
