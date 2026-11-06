function write_csv(file, cols, M)
% WRITE_CSV  写 UTF-8（带 BOM，Excel 可直接打开）CSV。cols 为列名元胞，M 为数值矩阵或元胞矩阵
fid = fopen(file, 'w');
fwrite(fid, uint8([239 187 191]));
fprintf(fid, '%s', cols{1}); for j = 2:numel(cols), fprintf(fid, ',%s', cols{j}); end; fprintf(fid, '\n');
for i = 1:size(M, 1)
    for j = 1:size(M, 2)
        if iscell(M), v = M{i, j}; else, v = M(i, j); end
        if j > 1, fprintf(fid, ','); end
        if ischar(v), fprintf(fid, '%s', v); else, fprintf(fid, '%.6g', v); end
    end
    fprintf(fid, '\n');
end
fclose(fid);
end
