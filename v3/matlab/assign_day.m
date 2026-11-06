function assign_day(V, R)
% ASSIGN_DAY  用已有运行解给本日决策变量赋初值（热启动）。只对 sdpvar/binvar 赋值，表达式跳过。
fn = {'mR','mB','UR','UB','PIT','LR','B','O','u','Tw','qFC','qMC','qHP','S','qch','qdis','z','PV','cur','SB','Pch','Pdis','b','buy','eps','eps_s'};
for i = 1:numel(fn)
    k = fn{i};
    if isfield(V, k) && isfield(R, k) && isequal(size(V.(k)), size(R.(k)))
        try, assign(V.(k), R.(k)); catch, end
    end
end
end
