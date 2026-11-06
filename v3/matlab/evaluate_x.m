function ev = evaluate_x(P, D, scen, x, th, a, int_add)
% EVALUATE_X  固定全部容量 x，逐典型日求运行成本，按最恶劣分布得目标值（式(32)在固定 x 下的取值）
%   ev.obj = Σ a_k x_k + 365·Σ p*_s Q_s ；a 由调用方给出（第二步 a.N = 0，目标即能源侧年化成本）
global CFG
if nargin < 7 || isempty(int_add), int_add = cell(numel(D.med), 1); end
K = numel(D.med); Q = zeros(K, 1); R = cell(K, 1);
usepar = ~isempty(CFG) && isfield(CFG, 'parallel') && CFG.parallel;
if usepar
    med = D.med;
    parfor s = 1:K
        [Q(s), R{s}] = solve_day_fixed(P, D, x, med(s), scen, int_add{s});
    end
else
    for s = 1:K
        [Q(s), R{s}] = solve_day_fixed(P, D, x, D.med(s), scen, int_add{s});
    end
end
p = worst_p(Q, D.p0, th(1), th(2));
inv = 0;
for i = 1:numel(P.CAPS), k = P.CAPS{i}; inv = inv + a.(k) * x.(k); end
ev = struct('x', x, 'Q', Q, 'R', {R}, 'p', p, 'inv', inv, 'obj', inv + P.DAYS * (p' * Q));
end
