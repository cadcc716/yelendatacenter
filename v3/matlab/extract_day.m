function R = extract_day(V)
% EXTRACT_DAY  取出单日全部变量与表达式的数值
fn = fieldnames(V); R = struct();
for i = 1:numel(fn)
    if strcmp(fn{i}, 'consts'), R.consts = V.consts; continue; end
    R.(fn{i}) = value(V.(fn{i}));
end
end
