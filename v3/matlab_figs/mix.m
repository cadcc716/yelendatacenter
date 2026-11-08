function s = mix(str)
% 中英文混排：汉字段用宋体，其余（英文、数字、符号）用 Times New Roman（TeX 解释器）。
s = ''; prev = -1; chars = char(str);
for i = 1:numel(chars)
    c = chars(i); isCJK = double(c) > 255;
    if isCJK ~= prev
        if isCJK, s = [s '\fontname{SimSun}']; else, s = [s '\fontname{Times New Roman}']; end
        prev = isCJK;
    end
    if any(c == '_^{}\'), s = [s '\' c]; else, s = [s c]; end
end
end
