function [imp, hist] = local_search(P, D, scen, th, best, a, max_eval)
% LOCAL_SEARCH  容量邻域搜索：主问题初始可行解改进。
%   在 x 邻域内做坐标搜索，每组候选容量逐典型日精确求运行问题，按式(37)得目标值，接受严格改进。
%   冷侧移动优先：E_TES↑且K_MC↓、E_TES↑且K_HP↑，再单变量 K_MC、E_TES、K_HP、K_FC、E_B、P_B。
%   最优性仍以 MILP 下界衡量。imp 为改进后的解（无改进时为空）。
if nargin < 7, max_eval = 50; end
fl = scen_flags(scen); C = model_consts(P); xm = C.xmax;
vars = {'MC', 'FC', 'EB', 'PB'};
if fl.tes, vars{end + 1} = 'TES'; end
if fl.heat, vars{end + 1} = 'HP'; end
step = struct('MC', 0.15, 'FC', 0.15, 'EB', 0.2, 'PB', 0.2, 'TES', 0.5, 'HP', 0.3);
flo = struct('MC', 50, 'FC', 100, 'EB', 1000, 'PB', 200, 'TES', 1000, 'HP', 100);
absmin = struct('MC', 0, 'FC', 0, 'EB', 0, 'PB', 0, 'TES', 5000, 'HP', 300);
cur = best.x; fcur = best.obj; keep = []; hist = []; n = 0; improved = true; t0 = tic;
while n < max_eval
    if ~improved
        small = true;
        for i = 1:numel(vars), k = vars{i}; small = small && step.(k) * max(cur.(k), absmin.(k)) < flo.(k); end
        if small, break; end
        for i = 1:numel(vars), step.(vars{i}) = step.(vars{i}) / 2; end
    end
    improved = false; moves = {};
    if fl.tes
        dt = max(step.TES * max(cur.TES, absmin.TES), flo.TES); dm = max(step.MC * cur.MC, flo.MC);
        moves = [moves, {struct('TES', dt, 'MC', -dm), struct('TES', 2 * dt, 'MC', -2 * dm), struct('TES', -dt, 'MC', dm)}];
    end
    if fl.heat && fl.tes
        moves{end + 1} = struct('HP', max(step.HP * max(cur.HP, absmin.HP), flo.HP), 'TES', max(step.TES * max(cur.TES, absmin.TES), flo.TES));
    end
    order = {'MC', 'TES', 'HP', 'FC', 'EB', 'PB'};
    for i = 1:numel(order)
        k = order{i}; if ~any(strcmp(vars, k)), continue; end
        d = max([step.(k) * cur.(k), step.(k) * absmin.(k), flo.(k)]);
        moves = [moves, {struct(k, d), struct(k, -d)}]; %#ok<AGROW>
    end
    for m = 1:numel(moves)
        mv = moves{m}; x = cur; fn = fieldnames(mv);
        for f = 1:numel(fn), x.(fn{f}) = x.(fn{f}) + mv.(fn{f}); end
        if isfield(mv, 'EB') && ~isfield(mv, 'PB'), x.PB = min(x.PB, P.C_B_MAX * x.EB); end
        ok = x.PB <= P.C_B_MAX * x.EB + 1e-6;
        for f = 1:numel(vars), ok = ok && x.(vars{f}) >= -1e-6 && x.(vars{f}) <= xm.(vars{f}) + 1e-6; end
        if ~ok, continue; end
        ev = evaluate_x(P, D, scen, x, th, a); n = n + 1;
        hist = [hist, struct('move', mv, 'obj', ev.obj)]; %#ok<AGROW>
        if ev.obj < fcur - 1
            log_line(sprintf('  [邻域搜索 %s] %d: %s → %.2f 万元（改进 %.2f），用时 %.0f s', scen, n, move_str(mv), ev.obj / 1e4, (fcur - ev.obj) / 1e4, toc(t0)));
            cur = x; fcur = ev.obj; keep = ev; improved = true; break;
        end
        if n >= max_eval, break; end
    end
end
if ~isempty(keep) && fcur < best.obj - 1, imp = keep; else, imp = []; end
end

function s = move_str(mv)
fn = fieldnames(mv); s = '';
for i = 1:numel(fn), s = [s, sprintf('%s%+.0f ', fn{i}, mv.(fn{i}))]; end %#ok<AGROW>
end
