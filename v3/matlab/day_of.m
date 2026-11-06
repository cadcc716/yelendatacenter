function d = day_of(D, i)
% DAY_OF  取第 i 天（1 起始）的日轨迹
d = struct('AR', D.AR(i, :), 'AB', D.AB(i, :), 'Tamb', D.Tamb(i, :), 'gPV', D.gPV(i, :), 'H', D.H(i, :));
end
