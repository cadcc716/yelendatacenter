function [val, lb, ok, info] = run_opt(F, obj, ops)
% RUN_OPT  调用 optimize，返回目标值、下界（Gurobi 的 best bound）与是否得到可行解。
diag = optimize(F, obj, ops);
val = value(obj); info = diag;
ok = ~isnan(val) && any(diag.problem == [0 3 4 5 9 15]) && isfinite(val);
lb = val;
try
    r = diag.solveroutput.result;
    if isfield(r, 'objbound') && isfield(r, 'objval') && ~isempty(r.objbound)
        lb = val - (r.objval - r.objbound);         % 消除 YALMIP 常数项偏移
    end
catch
end
if ~ok, val = NaN; end
end
