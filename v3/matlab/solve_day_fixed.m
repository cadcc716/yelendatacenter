function [Qv, R, ok] = solve_day_fixed(P, D, x, i, scen, int_add, gap, tl)
% SOLVE_DAY_FIXED  子问题(36)：固定全部容量 x，求第 i 天的运行 MILP
if nargin < 6 || isempty(int_add), int_add = struct(); end
if nargin < 7, gap = 1e-4; end
if nargin < 8, tl = 600; end
fl = scen_flags(scen);
[X, F0] = build_capacity(P, fl, 'fix', x);
[F, Q, V] = build_day(P, X, day_of(D, i), fl, tamb_design(D, P), x.N, int_add);
[Qv, ~, ok] = run_opt([F0, F], Q, solver_ops(gap, tl));
if ~ok, error('子问题求解失败：第 %d 天，场景 %s', i, scen); end
R = extract_day(V);
end
