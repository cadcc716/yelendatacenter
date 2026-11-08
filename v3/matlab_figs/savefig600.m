function savefig600(f, name)
% 600 dpi PNG，输出到 out/ 目录
if ~exist('out', 'dir'), mkdir('out'); end
exportgraphics(f, fullfile('out', [name '.png']), 'Resolution', 600, 'BackgroundColor', 'white');
end
