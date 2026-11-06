function ops = solver_ops(gap, tl, cutoff, usex0, show)
% SOLVER_OPS  求解器设置。默认 Gurobi；全局变量 CFG.solver 可改为 'glpk'（仅用于无 Gurobi 时的语法测试）。
%   gap: MIPGap（相对）；tl: 时间上限 s；cutoff: 截断值（可为空）；usex0: 是否用当前变量值热启动；
%   show: 是否在命令行窗口显示 Gurobi 求解日志（主问题与单层 MILP 缺省显示，子问题不显示）
global CFG
if isempty(CFG) || ~isfield(CFG, 'solver'), CFG.solver = 'gurobi'; end
if nargin < 3, cutoff = []; end
if nargin < 4, usex0 = 0; end
if nargin < 5, show = 0; end
if strcmp(CFG.solver, 'gurobi')
    ops = sdpsettings('solver', 'gurobi', 'verbose', double(show), 'usex0', usex0, 'cachesolvers', 1, ...
                      'savesolveroutput', 1);  % 保留 Gurobi 返回结构，用于读取 best bound（下界）
    ops.gurobi.DisplayInterval = 60;       % 日志每 60 s 刷新一次
    ops.gurobi.MIPGap = gap;
    ops.gurobi.TimeLimit = tl;
    ops.gurobi.Threads = 0;                 % 0 = 使用全部核
    if isfield(CFG, 'MIPFocus'), ops.gurobi.MIPFocus = CFG.MIPFocus; else, ops.gurobi.MIPFocus = 3; end
    if ~isempty(cutoff), ops.gurobi.Cutoff = cutoff; end
else
    ops = sdpsettings('solver', CFG.solver, 'verbose', 0, 'cachesolvers', 1);
end
end
