function run_cases(site, tags, method, varargin)
% RUN_CASES  批量求解 S1–S7（执行方案 5.2 节），两步法 + 嵌套热启动 + 邻域搜索
%   run_cases('zhongwei')                         全部 7 个场景，第二步用 C&CG
%   run_cases('zhongwei', {'S3','S4','S5'})       指定场景（嵌套链按 S1→S2、S3→S4→S5 顺序自动衔接）
%   run_cases('zhongwei', {}, 'mono')             第二步用单层 MILP(40)，结果文件名加 _mono，用于与 C&CG 对照（不做邻域搜索）
%   其余参数传给 ccg_two_step 的 opt，如 'tl2', 1800, 'ls_max', 30；参数覆盖用 'P', {'C_HEAT', 0.25}
%   结果：results/<site>_<DRO|SP|RO>_<Sx>[_mono].mat，变量 r
global CFG
if nargin < 2 || isempty(tags), tags = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'}; end
if nargin < 3 || isempty(method), method = 'ccg'; end
opt = struct(); pover = {};
for i = 1:2:numel(varargin)
    if strcmp(varargin{i}, 'P'), pover = varargin{i + 1}; else, opt.(varargin{i}) = varargin{i + 1}; end
end
if strcmp(method, 'mono') && ~isfield(opt, 'ls'), opt.ls = false; end
D = load(fullfile(CFG.data, [site '_data.mat']));
P = params_v3(pover{:}); P.K = numel(D.med);
jobs = struct('S1', {{'S1', 'DRO', ''}}, 'S2', {{'S2', 'DRO', 'S1'}}, 'S3', {{'S3', 'DRO', ''}}, ...
              'S4', {{'S4', 'DRO', 'S3'}}, 'S5', {{'S5', 'DRO', 'S4'}}, 'S6', {{'S5', 'SP', ''}}, 'S7', {{'S5', 'RO', ''}});
suffix = ''; if strcmp(method, 'mono'), suffix = '_mono'; end
for i = 1:numel(tags)
    tg = tags{i}; j = jobs.(tg); scen = j{1}; unc = j{2};
    switch unc
        case 'SP', th = [0 0];
        case 'RO', th = [2 1];
        otherwise, [a1, ai] = theta_radius(P, numel(D.med)); th = [a1 ai];
    end
    prev = [];
    if ~isempty(j{3})
        pj = jobs.(j{3}); f = fullfile(CFG.res, sprintf('%s_%s_%s%s.mat', site, pj{2}, j{3}, suffix));
        if exist(f, 'file'), S = load(f); prev = S.r; end
    end
    log_line(sprintf('==== %s %s（开关 %s，θ=(%.4f,%.4f)，第二步 %s）====', site, tg, scen, th(1), th(2), method));
    r = ccg_two_step(P, D, scen, th, method, prev, opt); r.P = P; r.site = site; r.tag = tg; %#ok<STRNU>
    save(fullfile(CFG.res, sprintf('%s_%s_%s%s.mat', site, unc, tg, suffix)), 'r', '-v7');
end
end
