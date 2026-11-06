function [th1, thi] = theta_radius(P, K)
% THETA_RADIUS  式(31)：综合范数模糊集半径。K 缺省取 P.K。
if nargin < 2, K = P.K; end
th1 = K / (2 * P.M) * log(2 * K / (1 - P.ALPHA1));
thi = 1 / (2 * P.M) * log(2 * K / (1 - P.ALPHAINF));
end
