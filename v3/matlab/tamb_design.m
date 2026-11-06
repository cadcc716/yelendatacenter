function Td = tamb_design(D, P)
% TAMB_DESIGN  设计气温：全年逐时气温的 P.AMB_DES_Q 分位数（线性插值，与 numpy.quantile 一致）
x = sort(D.Tamb(:)); n = numel(x); h = (n - 1) * P.AMB_DES_Q + 1;
Td = x(floor(h)) + (h - floor(h)) * (x(min(n, floor(h) + 1)) - x(floor(h)));
end
