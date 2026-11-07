function run_cases(site, tags, method, varargin)
% RUN_CASES  批量求解 S1–S7（执行方案 5.2 节），两步法 + 嵌套热启动 + 邻域搜索
%   run_cases('zhongwei')                         全部 7 个场景，第二步用 C&CG（主问题时限缺省 1200 s）
%   run_cases('zhongwei', {'S3','S4','S5'})       指定场景（嵌套链按 S1→S2、S3→S4→S5 顺序自动衔接）
%   run_cases('zhongwei', {}, 'mono')             第二步用单层 MILP(40)，结果文件名加 _mono，用于与 C&CG 对照（不做邻域搜索）
%   其余参数传给 ccg_two_step 的 opt，如 'tl2', 1800, 'ls_max', 20；参数覆盖用 'P', {'C_HEAT', 0.25}
%   断点续跑：已有且有效的结果直接跳过。下列情况自动重算：
%     (1) 旧版本（v1.2 之前，下界读取有误）生成的结果；
%     (2) 有前驱场景（S2←S1、S4←S3、S5←S4）、前驱结果已存在，但本场景当时未用前驱做嵌套热启动；
%   强制重算加 'force', true。单个场景出错时记录日志并继续后续场景。
global CFG
if nargin < 2 || isempty(tags), tags = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'}; end
if nargin < 3 || isempty(method), method = 'ccg'; end
opt = struct(); pover = {}; force = false;
for i = 1:2:numel(varargin)
    if strcmp(varargin{i}, 'P'), pover = varargin{i + 1};
    elseif strcmp(varargin{i}, 'force'), force = varargin{i + 1};
    else, opt.(varargin{i}) = varargin{i + 1}; end
end
if strcmp(method, 'mono') && ~isfield(opt, 'ls'), opt.ls = false; end
D = load(fullfile(CFG.data, [site '_data.mat']));
P = params_v3(pover{:}); P.K = numel(D.med);
jobs = struct('S1', {{'S1', 'DRO', ''}}, 'S2', {{'S2', 'DRO', 'S1'}}, 'S3', {{'S3', 'DRO', ''}}, ...
              'S4', {{'S4', 'DRO', 'S3'}}, 'S5', {{'S5', 'DRO', 'S4'}}, 'S6', {{'S5', 'SP', ''}}, 'S7', {{'S5', 'RO', ''}});
suffix = ''; if strcmp(method, 'mono'), suffix = '_mono'; end
for i = 1:numel(tags)
    tg = tags{i}; j = jobs.(tg); scen = j{1}; unc = j{2};
    fout = fullfile(CFG.res, sprintf('%s_%s_%s%s.mat', site, unc, tg, suffix));
    fprev = '';
    if ~isempty(j{3})
        pj = jobs.(j{3}); fprev = fullfile(CFG.res, sprintf('%s_%s_%s%s.mat', site, pj{2}, j{3}, suffix));
        if ~exist(fprev, 'file'), fprev = ''; end
    end
    if ~force && exist(fout, 'file')
        [ok, why] = result_valid(fout, ~isempty(fprev));
        if ok
            log_line(sprintf('%s %s 已有有效结果，跳过（重算请加 ''force'', true）', site, tg)); continue;
        end
        log_line(sprintf('%s %s 已有结果需重算：%s', site, tg, why));
    end
    switch unc
        case 'SP', th = [0 0];
        case 'RO', th = [2 1];
        otherwise, [a1, ai] = theta_radius(P, numel(D.med)); th = [a1 ai];
    end
    prev = [];
    if ~isempty(fprev), S = load(fprev); prev = S.r; end
    log_line(sprintf('==== %s %s（开关 %s，θ=(%.4f,%.4f)，第二步 %s）====', site, tg, scen, th(1), th(2), method));
    try
        r = ccg_two_step(P, D, scen, th, method, prev, opt); r.P = P; r.site = site; r.tag = tg; %#ok<STRNU>
        save(fout, 'r', '-v7');
    catch err
        log_line(sprintf('!!!! %s %s 求解出错，已跳过：%s', site, tg, err.message));
        for k = 1:min(3, numel(err.stack)), log_line(sprintf('     位置：%s 第 %d 行', err.stack(k).name, err.stack(k).line)); end
    end
end
end

function [ok, why] = result_valid(f, needprev)
% 结果有效性：v2 生成，或 v1.2–v1.4 生成且下界正常（gap > 0）；有前驱结果时须用过嵌套热启动
S = load(f); r = S.r; ok = true; why = '';
if ~isfield(r, 'ver') && ~(isfield(r, 'gap_final') && r.gap_final > 1e-6)
    ok = false; why = '旧版本生成（下界读取有误）'; return;
end
if needprev && ~(isfield(r, 'used_prev') && r.used_prev)
    ok = false; why = '前驱场景结果已存在，但本场景当时未做嵌套热启动（需保证能源侧成本单调）';
end
end
