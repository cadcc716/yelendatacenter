function r = solve_dro(P, D, scen, th, mode, Nval, method, gap, tl, seed, int_add, opt)
% SOLVE_DRO  求解式(32)：method = 'ccg'（主问题(35)+子问题(36)(37)）或 'mono'（等价单层 MILP(40)）
%   mode = 'free'（第一步，总成本，N 连续）或 'fixN'（第二步，N = Nval，目标为能源侧年化成本）
%   seed : 已有可行解（上一场景或第一步容量的评估结果），用作初始解、截断值与上界初值；可为空
%   返回 r：x, obj, inv, Q, p, R, LB, gap_final, hist, t
if nargin < 10, seed = []; end
if nargin < 11 || isempty(int_add), int_add = cell(numel(D.med), 1); end
if nargin < 12, opt = struct(); end
if ~isfield(opt, 'eps'), opt.eps = 1e-3; end
if ~isfield(opt, 'max_it'), opt.max_it = 10; end
fl = scen_flags(scen); K = numel(D.med); Td = tamb_design(D, P); t0 = tic;
[X, F, inv, a] = build_capacity(P, fl, mode, Nval);
if strcmp(mode, 'fixN'), Nfix = Nval; else, Nfix = []; end
Qs = cell(K, 1); Vs = cell(K, 1);
for s = 1:K
    [Fs, Qs{s}, Vs{s}] = build_day(P, X, day_of(D, D.med(s)), fl, Td, Nfix, int_add{s});
    F = [F, Fs];
end
Qv = [Qs{:}]';                                   % K×1 表达式
cutoff = [];
if ~isempty(seed)
    cutoff = seed.obj * (1 + 1e-7) + 1;
    warm_start(X, Vs, seed, P);
end
hist = []; LB = -inf; UB = inf; best = seed;
if ~isempty(seed), UB = seed.obj; end
if strcmp(method, 'mono')
    lam = sdpvar(1); pi_ = sdpvar(K, 1); mu = sdpvar(1); nu = sdpvar(K, 1);
    Fm = [F, lam + pi_ >= P.DAYS * Qv, pi_ <= mu + nu, pi_ >= -(mu + nu), mu >= 0, nu >= 0];
    obj = inv + lam + D.p0(:)' * pi_ + th(1) * mu + th(2) * sum(nu);
    if ~isempty(seed), assign(lam, P.DAYS * max(seed.Q)); assign(pi_, zeros(K, 1)); assign(mu, 0); assign(nu, zeros(K, 1)); end
    [val, lb, ok] = run_opt(Fm, obj, solver_ops(gap, tl, cutoff, ~isempty(seed)));
    LB = lb;
    if ok
        ev = evaluate_x(P, D, scen, xval(X, P), th, a, int_add);
        if ev.obj < UB, UB = ev.obj; best = ev; end
    end
    hist = struct('it', 1, 'LB', LB, 'UB', UB, 'gap', (UB - LB) / abs(UB), 't', toc(t0), 'milp', val);
    log_line(sprintf('[%s 单层MILP θ=(%.3g,%.3g) %s] MILP %.2f 万元, bound %.2f, 最终 %.2f 万元, gap %.2e, 用时 %.0f s', ...
        scen, th(1), th(2), mode, val / 1e4, LB / 1e4, UB / 1e4, (UB - LB) / abs(UB), toc(t0)));
else
    eta = sdpvar(1); cuts = []; p = D.p0(:); Fc = F;
    for k = 1:opt.max_it
        cuts = [cuts, p]; %#ok<AGROW>
        Fc = [Fc, eta >= P.DAYS * (p' * Qv)];
        if k > 1 || ~isempty(seed)          % η 取满足全部割的最小值，保证热启动可行
            qv = value(Qv); assign(eta, max(P.DAYS * (cuts' * qv)));
        end
        [val, lb, ok] = run_opt(Fc, inv + eta, solver_ops(gap, tl, cutoff, k > 1 || ~isempty(seed)));
        if ~ok
            log_line(sprintf('[%s] 主问题未得到优于截断值的解，沿用初始解', scen)); break;
        end
        LB = max(LB, lb);
        ev = evaluate_x(P, D, scen, xval(X, P), th, a, int_add);
        if ev.obj < UB, UB = ev.obj; best = ev; end
        p = worst_p(ev.Q, D.p0, th(1), th(2));
        g = (UB - LB) / abs(UB);
        hist = [hist, struct('it', k, 'LB', LB, 'UB', UB, 'gap', g, 't', toc(t0), 'milp', val)]; %#ok<AGROW>
        x = ev.x;
        log_line(sprintf('[%s θ=(%.3g,%.3g) %s] 迭代 %d: LB %.2f, UB %.2f 万元, gap %.2e, N %.1f, FC %.0f, MC %.0f, TES %.0f, HP %.0f, PV %.0f, EB %.0f, PB %.0f, 用时 %.0f s', ...
            scen, th(1), th(2), mode, k, LB / 1e4, UB / 1e4, g, x.N, x.FC, x.MC, x.TES, x.HP, x.PV, x.EB, x.PB, toc(t0)));
        if g <= opt.eps, break; end
        if any(max(abs(cuts - repmat(p, 1, size(cuts, 2))), [], 1) < 1e-6), break; end   % 最恶劣分布重复，判定收敛
    end
end
r = best; r.LB = LB; r.gap_final = (r.obj - LB) / abs(r.obj); r.hist = hist; r.t = toc(t0); r.a = a; r.method = method;
end

function x = xval(X, P)
x = struct();
for i = 1:numel(P.CAPS)
    k = P.CAPS{i};
    if isa(X.(k), 'sdpvar'), x.(k) = value(X.(k)); else, x.(k) = X.(k); end
    if abs(x.(k)) < 1e-6, x.(k) = 0; end
end
end

function warm_start(X, Vs, seed, P)
for i = 1:numel(P.CAPS)
    k = P.CAPS{i};
    if isa(X.(k), 'sdpvar'), assign(X.(k), seed.x.(k)); end
end
for s = 1:numel(Vs), assign_day(Vs{s}, seed.R{s}); end
end
