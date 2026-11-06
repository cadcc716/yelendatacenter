function [F, Q, V] = build_day(P, X, day, fl, Tamb_des, Nfix, int_add)
% BUILD_DAY  单个典型日的运行约束 式(1)–(29)（与 Python 版 build_op.add_day 逐条对应）
%   X        : build_capacity 返回的容量（sdpvar 或数值）
%   day      : 结构体，字段 AR AB Tamb gPV H（均 1×T）
%   Tamb_des : 设计气温（S1/S2 冷机 COP 设计日值用）
%   Nfix     : GPU 数固定时的数值（式(12)大M取实际装机数）；为空则取 P.N_BIGM
%   int_add  : 结构体，字段 b / z 为需恢复 0-1 的时段（校验同时充放后使用），可缺省
%   F 为约束，Q 为该日运行成本（元/日，式(29)），V 为全部变量（便于取值与热启动）
if nargin < 6, Nfix = []; end
if nargin < 7, int_add = struct(); end
T = P.T; dt = P.DT; G = P.GEARS; KR = P.K_R; KB = P.K_B; nR = numel(KR); nB = numel(KB);
AR = day.AR(:)'; AB = day.AB(:)'; Ta = day.Tamb(:)'; gpv = day.gPV(:)'; Hd = day.H(:)';
C = model_consts(P, Ta); J = C.J; xm = C.xmax;
if isempty(Nfix), Nmax = P.N_BIGM; else, Nmax = min(P.N_BIGM, Nfix); end

% ---------- 温区相关常数 ----------
if fl.thermal
    phi = C.phi; cop = C.cop; plk = C.plk; safe = C.safe; zone_ok = true(J, 1);
else
    jd = find(C.lo == P.T_W_DES, 1);
    phi = C.phi;                                         % 自然冷却：45–50 ℃ 温区、逐时气温（只有该温区可用）
    Cd = model_consts(P, Tamb_des); cop = repmat(Cd.cop, 1, T);   % 冷机 COP：设计日值
    plk = P.P_LK0 * ones(J, 1); safe = true(size(G, 1), J);
    zone_ok = false(J, 1); zone_ok(jd) = true;
end
if fl.dvfs, gear_ok = true(size(G, 1), 1); else, gear_ok = false(size(G, 1), 1); gear_ok(1) = true; end
okR = repmat(gear_ok(KR), 1, J) & safe(KR, :) & repmat(zone_ok', nR, 1);    % nR×J
okB = repmat(gear_ok(KB), 1, J) & safe(KB, :) & repmat(zone_ok', nB, 1);

% 行编号：第 (i-1)*J + j 行 = 第 i 个档位、第 j 个温区
idxR = @(i, j) (i - 1) * J + j;
okRv = reshape(okR', [], 1); okBv = reshape(okB', [], 1);
vR = kron(G(KR, 2), ones(J, 1)); vB = kron(G(KB, 2), ones(J, 1));                       % 吞吐比
pR = kron(G(KR, 3), ones(J, 1)) + repmat(plk, nR, 1);                                   % 动态 + 漏电
pB = kron(G(KB, 3), ones(J, 1)) + repmat(plk, nB, 1);
zR = repmat((1:J)', nR, 1); zB = repmat((1:J)', nB, 1);                                  % 各行所在温区

% ---------- (1)(2)(3) GPU 分配与功率 ----------
V.mR = sdpvar(nR * J, T, 'full'); V.mB = sdpvar(nB * J, T, 'full');
F = [V.mR >= 0, V.mB >= 0];
F = [F, V.mR(~okRv, :) == 0, V.mB(~okBv, :) == 0, V.mR <= Nmax, V.mB <= Nmax];
F = [F, sum(V.mR, 1) + sum(V.mB, 1) <= X.N];
V.UR = sdpvar(1, T); V.UB = sdpvar(1, T);
F = [F, V.UR >= 0, V.UB >= 0, V.UR <= dt * vR' * V.mR, V.UB <= dt * vB' * V.mB];
V.PIT = sdpvar(1, T);
F = [F, V.PIT == X.N * P.P_BASE + pR' * V.mR + pB' * V.mB];

% ---------- (5)–(8) 算力任务 ----------
V.LR = sdpvar(1, T); V.B = sdpvar(1, T + 1); V.O = sdpvar(1, T);
F = [F, 0 <= V.LR <= AR, V.UR + V.LR == AR];
F = [F, V.B >= 0, V.B(2:end) == V.B(1:T) + AB - V.UB, V.O >= 0];
D = P.D_DEADLINE; cumA = cumsum(AB); rhs = zeros(1, T); cumU = cumsum(V.UB);
if P.B_PERIODIC
    % 式(7)(8)：积压周期条件 B_1 = B_{T+1}；先到先服务，结转积压由上一周期最后到达的任务构成
    F = [F, V.B(T + 1) == V.B(1)];
    for t = 1:T
        if t >= D + 1
            rhs(t) = cumA(t - D);
        elseif t <= D - 1
            rhs(t) = -sum(AB(T + t - D + 1:T));
        end
    end
    F = [F, V.O + cumU - V.B(1) >= rhs];
else
    F = [F, V.B(1) == 0, V.B(T + 1) == 0];
    for t = D + 1:T, rhs(t) = cumA(t - D); end
    F = [F, V.O(D + 1:T) + cumU(D + 1:T) >= rhs(D + 1:T)];
end

% ---------- (9)(10) 液冷热状态与温区 ----------
V.u = binvar(J, T, 'full');
if fl.thermal
    F = [F, sum(V.u, 1) == 1];
else
    F = [F, V.u(jd, :) == 1, V.u(~zone_ok, :) == 0];
end
V.Tw = sdpvar(1, T + 1);
F = [F, min(C.lo) <= V.Tw <= max(C.hi), V.Tw(1:T) >= C.lo' * V.u, V.Tw(1:T) <= C.hi' * V.u, V.Tw(T + 1) == V.Tw(1)];
nxt = [2:T, 1];

% ---------- (12) 热安全门控 ----------
for j = 1:J        % 第一行按温区汇总：Σ_{r,k} m_rkjt ≤ N^max·u_jt
    rR = find(okRv & zR == j); rB = find(okBv & zB == j);
    if ~isempty(rR) || ~isempty(rB)
        F = [F, sum(V.mR(rR, :), 1) + sum(V.mB(rB, :), 1) <= Nmax * V.u(j, :)];
    end
end
if fl.thermal && P.GATE_NEXT   % 第二行：时段末温区对所用档位同样安全
    for i = 1:nR
        sj = find(safe(KR(i), :));
        F = [F, sum(V.mR(idxR(i, 1):idxR(i, J), :), 1) <= Nmax * sum(V.u(sj, nxt), 1)];
    end
    for i = 1:nB
        sj = find(safe(KB(i), :));
        F = [F, sum(V.mB(idxR(i, 1):idxR(i, J), :), 1) <= Nmax * sum(V.u(sj, nxt), 1)];
    end
end

% ---------- (13)–(16) 自然冷却与机械制冷 ----------
V.qFC = sdpvar(J, T, 'full'); V.qMC = sdpvar(J, T, 'full'); V.qHP = sdpvar(J, T, 'full');
F = [F, V.qFC >= 0, V.qMC >= 0, V.qHP >= 0];
F = [F, V.qFC <= phi * X.FC, V.qFC <= (phi * xm.FC) .* V.u];
F = [F, V.qMC <= xm.MC * V.u, V.qHP <= xm.HP * V.u, sum(V.qMC, 1) <= X.MC];
% 自然冷却容量按温区汇总（有效不等式）：Σ_j q_FC,jt / φ_jt ≤ K_FC
W = zeros(J, T); pos = phi > 1e-6; W(pos) = 1 ./ phi(pos);
cols = find(sum(pos, 1) > 1);
if ~isempty(cols)
    F = [F, sum(W(:, cols) .* V.qFC(:, cols), 1) <= X.FC];
end
V.PFC = P.A_FC * sum(V.qFC, 1);
V.PMC = sum((1 ./ cop) .* V.qMC, 1);

% ---------- (17) 蓄冷罐 ----------
V.S = sdpvar(1, T + 1); V.qch = sdpvar(1, T); V.qdis = sdpvar(1, T);
if fl.tes && ~P.RELAX_COMPL, V.z = binvar(1, T); else, V.z = sdpvar(1, T); end
Qt = xm.TES / P.TES_TAU;
F = [F, V.S >= 0, V.qch >= 0, V.qdis >= 0, 0 <= V.z <= double(fl.tes)];
F = [F, V.S(2:end) == (1 - P.TES_LOSS) * V.S(1:T) + P.TES_EFF * dt * V.qch - dt / P.TES_EFF * V.qdis];
F = [F, V.S <= X.TES, V.qch <= X.TES / P.TES_TAU, V.qdis <= X.TES / P.TES_TAU];
F = [F, V.qch <= Qt * V.z, V.qdis <= Qt * (1 - V.z)];
F = [F, V.S(1) == P.TES_SOC0 * X.TES, V.S(T + 1) == P.TES_SOC0 * X.TES];

% ---------- (9) 液冷环路热平衡（C_w 归入热耦合） ----------
if fl.thermal, Cw = P.C_W; else, Cw = 0; end
F = [F, Cw * (V.Tw(2:end) - V.Tw(1:T)) == dt * (P.BETA * V.PIT - sum(V.qFC + V.qMC + V.qHP, 1) + V.qch - V.qdis)];

% ---------- (18)(19)(25) 余热热泵 ----------
ch = C.cop_hp;
V.PHP = (1 ./ (ch - 1))' * V.qHP;
V.HHP = (ch ./ (ch - 1))' * V.qHP;
F = [F, V.HHP <= Hd, V.HHP <= X.HP];

% ---------- (20)–(23) 辅助电耗、光伏、储能、电网（防逆流） ----------
V.PV = sdpvar(1, T); V.cur = sdpvar(1, T);
F = [F, V.PV >= 0, V.cur >= 0, V.PV <= gpv * X.PV, V.cur == gpv * X.PV - V.PV];
V.SB = sdpvar(1, T + 1); V.Pch = sdpvar(1, T); V.Pdis = sdpvar(1, T);
if P.RELAX_COMPL, V.b = sdpvar(1, T); else, V.b = binvar(1, T); end
PBm = xm.PB;
F = [F, V.Pch >= 0, V.Pdis >= 0, 0 <= V.b <= 1];
F = [F, V.SB(2:end) == (1 - P.B_LOSS) * V.SB(1:T) + P.B_ETA_C * dt * V.Pch - dt / P.B_ETA_D * V.Pdis];
F = [F, V.SB <= P.B_SOC_MAX * X.EB, V.SB >= P.B_SOC_MIN * X.EB, V.Pch <= X.PB, V.Pdis <= X.PB];
F = [F, V.Pch <= PBm * V.b, V.Pdis <= PBm * (1 - V.b)];
F = [F, V.SB(1) == P.B_SOC0 * X.EB, V.SB(T + 1) == P.B_SOC0 * X.EB];
V.buy = sdpvar(1, T);
F = [F, 0 <= V.buy <= P.P_GRID_MAX];
% 恢复 0-1：只在校验发现同时充放的时段
if isfield(int_add, 'b') && ~isempty(int_add.b)
    bb = binvar(1, numel(int_add.b)); F = [F, V.b(int_add.b) == bb];
end
if isfield(int_add, 'z') && ~isempty(int_add.z)
    zz = binvar(1, numel(int_add.z)); F = [F, V.z(int_add.z) == zz];
end

% ---------- (24) 电功率平衡 ----------
F = [F, V.buy + V.PV + V.Pdis == (1 + C.kaux) * V.PIT + V.PFC + V.PMC + V.PHP + V.Pch + P.P0];

% ---------- (26)(27) 阶梯碳交易 ----------
Vn = P.CO2_V;
V.eps = sdpvar(Vn, 1); V.eps_s = sdpvar(1, 1);
F = [F, V.eps >= 0, V.eps(1:Vn - 1) <= P.CO2_LEN, V.eps_s >= 0];
F = [F, P.E_GRID * dt * sum(V.buy) - P.E_HEAT * dt * sum(V.HHP) - P.QUOTA * (sum(V.UR) + sum(V.UB)) == sum(V.eps) - V.eps_s];

% ---------- (29) 运行成本 ----------
lam = P.CO2_PRICE;
Q = dt * (C.price * V.buy') - P.C_HEAT * dt * sum(V.HHP) + P.C_CUR * dt * sum(V.cur) ...
    + P.C_DEG * dt * sum(V.Pch + V.Pdis) + P.C_R * sum(V.LR) + P.C_O * sum(V.O) ...
    + lam * (1 + (0:Vn - 1) * P.CO2_GROW) * V.eps - lam * V.eps_s;
V.consts = struct('phi', phi, 'cop', cop, 'plk', plk, 'kaux', C.kaux, 'price', C.price, 'nR', nR, 'nB', nB, 'J', J);
end
