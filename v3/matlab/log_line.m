function log_line(msg)
% LOG_LINE  屏幕输出并追加到日志文件（全局 CFG.logfile）
global CFG
fprintf('%s  %s\n', datestr(now, 'HH:MM:SS'), msg);
if ~isempty(CFG) && isfield(CFG, 'logfile') && ~isempty(CFG.logfile)
    fid = fopen(CFG.logfile, 'a');
    if fid > 0, fprintf(fid, '%s  %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), msg); fclose(fid); end
end
end
