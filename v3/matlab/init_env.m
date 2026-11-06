function init_env(varargin)
% INIT_ENV  路径与求解器设置。在 MATLAB 命令行先运行一次：
%   init_env('yalmip', 'D:\YALMIP-master')                 % 指定 YALMIP 目录（若已在路径中可省略）
%   init_env('solver', 'gurobi', 'parallel', false)        % 缺省即为 gurobi、串行
% 可选字段：solver（gurobi | glpk）、parallel（true 时子问题用 parfor）、MIPFocus（缺省 3）、logfile、
%           show_log（缺省 true：主问题与单层 MILP 求解时在命令行显示 Gurobi 日志）
global CFG
CFG = struct('solver', 'gurobi', 'parallel', false, 'MIPFocus', 3, 'logfile', '', 'show_log', true);
here = fileparts(mfilename('fullpath'));
for i = 1:2:numel(varargin)
    k = varargin{i}; v = varargin{i + 1};
    if strcmp(k, 'yalmip'), addpath(genpath(v)); else, CFG.(k) = v; end
end
addpath(here);
CFG.root = here; CFG.data = fullfile(here, 'data'); CFG.res = fullfile(here, 'results');
if ~exist(CFG.res, 'dir'), mkdir(CFG.res); end
if isempty(CFG.logfile), CFG.logfile = fullfile(CFG.res, 'run_log.txt'); end
if exist('sdpvar', 'file') ~= 2 && exist('sdpvar', 'class') == 0
    warning('未找到 YALMIP：请用 init_env(''yalmip'', ''YALMIP 所在目录'')');
end
log_line(sprintf('环境就绪：求解器 %s，并行 %d，结果目录 %s', CFG.solver, CFG.parallel, CFG.res));
end
