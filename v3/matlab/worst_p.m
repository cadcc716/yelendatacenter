function p = worst_p(Q, p0, th1, thi)
% WORST_P  子问题(37)：在综合范数模糊集内求最恶劣分布（线性规划）
%   max Σ p_s Q_s  s.t. Σp=1, p−p0 = p⁺−p⁻, Σ(p⁺+p⁻) ≤ θ1, p⁺_s+p⁻_s ≤ θ∞, p,p⁺,p⁻ ≥ 0
Q = Q(:); p0 = p0(:); K = numel(Q);
if th1 <= 0 && thi <= 0, p = p0; return; end
p = sdpvar(K, 1); pp = sdpvar(K, 1); pm = sdpvar(K, 1);
F = [sum(p) == 1, p - p0 == pp - pm, p >= 0, pp >= 0, pm >= 0, sum(pp + pm) <= th1, pp + pm <= thi];
optimize(F, -Q' * p, solver_ops(1e-9, 60));
p = max(value(p), 0); p = p / sum(p);
end
