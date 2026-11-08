# MATLAB 重绘脚本（SimSun + Times New Roman）

1. 先在 Python 侧运行 `code/plot_v4.py`，生成 `figures_v4/data/*.csv`（每张图的数据）。
2. 在 MATLAB（R2020b 及以上）中进入本目录，逐个运行 `fig*.m`，图片输出到 `out/`，600 dpi PNG。
3. 样式：单栏 8.5 cm；坐标标题 10 pt、刻度与图例 9 pt；主线宽 1.1 pt；Marker 4.8 pt；白底；图例无边框。
   中英文混排用 `mix()` 自动切换字体：汉字为宋体（SimSun），其余为 Times New Roman。
4. 已提供：图5（逐级分解）、图7（相对 Baseline 变化率，分组柱状图模板）、图9（典型日双子图模板）。
   其余各图的数据列与 Python 版一一对应，可按模板改写；Python 版 PNG 位于 `figures_v4/`（替代字体，度量与 Times New Roman/宋体一致）。
