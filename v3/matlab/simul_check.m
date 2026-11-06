function [found, int_add] = simul_check(R, int_add, tol)
% SIMUL_CHECK  检查同时充放电（b）、同时充放冷（z），返回需恢复 0-1 变量的时段
if nargin < 2 || isempty(int_add), int_add = cell(numel(R), 1); end
if nargin < 3, tol = 1.0; end
found = false;
for s = 1:numel(R)
    if isempty(int_add{s}), int_add{s} = struct('b', [], 'z', []); end
    tb = find(min(R{s}.Pch, R{s}.Pdis) > tol); tz = find(min(R{s}.qch, R{s}.qdis) > tol);
    if ~isempty(tb), int_add{s}.b = union(int_add{s}.b, tb); found = true; end
    if ~isempty(tz), int_add{s}.z = union(int_add{s}.z, tz); found = true; end
end
end
