function climate_compare(sites)
% CLIMATE_COMPARE  气候区对比（C13、图14）：各地 S5 的冷源/蓄冷/热泵配置、PUE、自然冷却利用小时、余热回收
%   climate_compare({'zhongwei', 'guian', 'shaoguan'})   需各地已运行 run_cases 与 summarize
%   结果：results/气候区对比.csv
global CFG
if nargin < 1, sites = {'zhongwei', 'guian', 'shaoguan'}; end
rows = {};
for i = 1:numel(sites)
    f = fullfile(CFG.res, [sites{i} '_tableE.csv']);
    if ~exist(f, 'file'), log_line(['缺少 ' f '，先运行 summarize']); continue; end
    fid = fopen(f, 'r'); fread(fid, 3); hdr = strsplit(strtrim(fgetl(fid)), ','); L = {};
    while true
        ln = fgetl(fid); if ~ischar(ln), break; end
        L(end + 1, :) = strsplit(strtrim(ln), ','); %#ok<AGROW>
    end
    fclose(fid);
    for k = 1:size(L, 1)
        if ~any(strcmp(L{k, 1}, {'S1', 'S3', 'S5'})), continue; end
        g = @(c) L{k, find(strcmp(hdr, c), 1)};
        rows(end + 1, :) = {sites{i}, L{k, 1}, g('K_FC/MW'), g('K_MC/MW'), g('E_TES/MWh'), g('K_HP/MW'), g('名义PUE'), ...
            g('自然冷却利用小时/h'), g('余热回收/MWh'), g('ERF/%'), g('能源侧年化成本/万元'), g('碳排放/t')}; %#ok<AGROW>
    end
end
write_csv(fullfile(CFG.res, '气候区对比.csv'), {'站点', '场景', 'K_FC/MW', 'K_MC/MW', 'E_TES/MWh', 'K_HP/MW', '名义PUE', ...
    '自然冷却利用小时/h', '余热回收/MWh', 'ERF/%', '能源侧年化成本/万元', '碳排放/t'}, rows);
log_line('气候区对比表已写入 results/气候区对比.csv');
end
