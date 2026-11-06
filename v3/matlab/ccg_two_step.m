function r = ccg_two_step(P, D, scen, th, method, prev, opt)
% CCG_TWO_STEP  两步法 + 嵌套热启动 + 容量邻域搜索（第 4.6 节）
%   第一步：总成本为目标，N 取连续变量，求 N*（取整）
%   第二步：固定 N = N*，只以能源侧年化成本为目标；MIPGap 与 ε 均相对能源侧成本
%   嵌套热启动：上一场景结果 prev（S4←S3、S5←S4、S2←S1，N* 相同时）与第一步容量取优，
%              作为初始解、截断值（Cutoff）与上界初值，保证能源侧成本单调
%   method：'ccg'（第二步用 C&CG）或 'mono'（第二步用单层 MILP(40)）
%   opt 字段：step1 ('mono'|'ccg'), gap1, tl1, gap2, tl2, eps, ls (true/false), ls_max
if nargin < 6, prev = []; end
d = struct('step1', 'mono', 'gap1', 1e-3, 'tl1', 900, 'gap2', 1e-3, 'tl2', 3600, 'eps', 1e-3, 'ls', true, 'ls_max', 50, 'max_it', 10);
fn = fieldnames(d);
for i = 1:numel(fn), if ~isfield(opt, fn{i}), opt.(fn{i}) = d.(fn{i}); end, end
t0 = tic;
% ---------- 第一步 ----------
r1 = solve_dro(P, D, scen, th, 'free', [], opt.step1, opt.gap1, opt.tl1, [], [], opt);
Nstar = round(r1.x.N);
log_line(sprintf('[%s] 第一步：N* = %d，总成本 %.2f 万元', scen, Nstar, r1.obj / 1e4));
% ---------- 第二步候选初值 ----------
C = model_consts(P); a2 = C.a; a2.N = 0;
x1 = r1.x; x1.N = Nstar;
seed = evaluate_x(P, D, scen, x1, th, a2);
log_line(sprintf('[%s] 第一步容量在第二步口径下的能源侧成本 %.2f 万元', scen, seed.obj / 1e4));
if ~isempty(prev)
    if abs(prev.x.N - Nstar) < 0.5
        pv = evaluate_x(P, D, scen, prev.x, th, a2);       % 上一场景解在本场景可行（新增设备取 0）
        log_line(sprintf('[%s] 上一场景解在本场景的能源侧成本 %.2f 万元', scen, pv.obj / 1e4));
        if pv.obj < seed.obj, seed = pv; end
    else
        log_line(sprintf('[%s] 上一场景 N*=%d 与本场景 N*=%d 不同，不做嵌套热启动', scen, prev.x.N, Nstar));
    end
end
% ---------- 第二步 ----------
int_add = [];
for rnd = 1:3
    r2 = solve_dro(P, D, scen, th, 'fixN', Nstar, method, opt.gap2, opt.tl2, seed, int_add, opt);
    if r2.obj > seed.obj * (1 + 1e-6)                        % 单调性校验（理论上不会出现，因 seed 为上界初值）
        log_line(sprintf('[%s] 第二步结果高于初值，延长时限重解', scen));
        r2b = solve_dro(P, D, scen, th, 'fixN', Nstar, method, opt.gap2, 2 * opt.tl2, seed, int_add, opt);
        if r2b.obj < r2.obj, r2 = r2b; end
    end
    r2.obj_pre_ls = r2.obj; r2.x_pre_ls = r2.x;
    if opt.ls
        [imp, lsh] = local_search(P, D, scen, th, r2, a2, opt.ls_max);
        r2.ls_hist = lsh;
        if ~isempty(imp)
            log_line(sprintf('[%s] 邻域搜索改进：%.2f → %.2f 万元', scen, r2.obj / 1e4, imp.obj / 1e4));
            keep = {'LB', 'hist', 'method', 'a', 'obj_pre_ls', 'x_pre_ls', 'ls_hist'};
            for i = 1:numel(keep), imp.(keep{i}) = r2.(keep{i}); end
            r2 = imp; r2.gap_final = (r2.obj - r2.LB) / abs(r2.obj);
        end
    end
    [found, int_add] = simul_check(r2.R, int_add);
    if ~found, break; end
    log_line(sprintf('[%s] 发现同时充放，恢复相应时段 0-1 变量后重解', scen));
end
r = r2; r.int_add = int_add; r.scen = scen; r.theta = th; r.days = D.med; r.p0 = D.p0; r.a_full = C.a;
r.step1 = struct('N', Nstar, 'obj_total', r1.obj, 'x', r1.x, 'gap', r1.gap_final, 't', r1.t);
r.t_total = toc(t0);
log_line(sprintf('[%s] 第二步：能源侧年化成本 %.2f 万元，gap %.2e，总用时 %.0f s', scen, r.obj / 1e4, r.gap_final, r.t_total));
end
