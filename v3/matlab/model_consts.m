function C = model_consts(P, Tamb)
% MODEL_CONSTS  预计算常数：漏电式(4)、安全温区式(11)、热泵COP式(18)、年化单价式(28)、分时电价；
%   给出 Tamb（1×T）时另算自然冷却系数式(13)与冷机COP式(16)，均为 J×T。
lo = P.ZONES(:,1); hi = P.ZONES(:,2); mid = (lo + hi) / 2;
C.lo = lo; C.hi = hi; C.mid = mid; C.J = numel(lo);
C.plk = P.P_LK0 * (1 + P.ALPHA_LK * (hi - P.T_REF_LK));                       % J×1
h = P.GEARS(:,4);
C.safe = (repmat(hi', numel(h), 1) + P.R_TH * repmat(h, 1, numel(lo))) <= P.T_CHIP_MAX + 1e-9;   % 档位×温区
C.cop_hp = min(P.COP_HP_MAX, P.ETA_C * (P.T_SUP + 273.15) ./ (P.T_SUP - mid));  % J×1
% 年化单价（元/单位/年）
C.a = struct();
for i = 1:numel(P.CAPS)
    k = P.CAPS{i}; c = P.COST.(k); r = P.DISC; n = c(2);
    C.a.(k) = r * (1 + r)^n / ((1 + r)^n - 1) * c(1) + c(3) * c(1);
    C.xmax.(k) = c(4);
end
% 分时电价（1×T，时段 t=1..24 对应 0–23 时）
pr = P.PRICE_FLAT * ones(1, P.T);
pr(P.PEAK_H + 1) = P.PRICE_PEAK; pr(P.VALLEY_H + 1) = P.PRICE_VALLEY;
C.price = pr;
C.kaux = P.DELTA_E + P.A_CIRC * P.BETA + P.A_AIR * (1 - P.BETA);
if nargin > 1
    Ta = repmat(Tamb(:)', C.J, 1); L = repmat(lo, 1, numel(Tamb));
    C.phi = min(1, max(0, (L - P.DT_APP - Ta) / P.DT_DES));
    C.cop = min(P.COP_MAX, max(P.COP_MIN, P.COP0 - P.C1 * (Ta - P.T_AMB_REF) + P.C2 * (L - P.T_W_REF)));
end
end
