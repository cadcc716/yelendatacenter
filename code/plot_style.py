"""
绘图样式
 style_a(): 时序对比图 —— 全部文字 23 pt; 中英文/数字/公式 Times New Roman (中文字形缺失时回退宋体);
            四边封闭; 左右轴线 1.5 pt, 上下轴线 1.4 pt; 主刻度线宽 1.35 pt, 长 7 pt; 横轴从 0 开始。
 style_b(): 重设计图 —— 中文宋体, 英文数字 Times New Roman; 坐标标题 10 pt, 刻度与图例 9 pt;
            主线宽 1.1 pt, Marker 4.5–5 pt; 白底, 图例无边框; 单栏宽 8.5 cm; 600 dpi PNG。
本机若无 Times New Roman / SimSun, 自动使用度量兼容的 Liberation Serif / Noto Serif CJK SC。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import numpy as np

_have = {f.name for f in fm.fontManager.ttflist}
LATIN0 = next(f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in _have)
CJK = next((f for f in ["SimSun", "Songti SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei"] if f in _have), None)


def _build_mixed_font():
    """matplotlib 的 mathtext 不做 CJK 字形回退, "时段/$t$" 这类中英混排公式会丢字。
    这里把 Times New Roman(或度量兼容的 Liberation Serif)与 宋体(或 Noto Serif CJK)按绘图脚本实际
    用到的汉字子集合并为一个字体 "SciMix Serif": 西文/数字/公式字形来自 Times, 汉字字形来自宋体。"""
    import re, glob, os
    from pathlib import Path
    from fontTools.ttLib import TTFont, newTable
    from fontTools import subset
    from fontTools.merge import Merger
    from fontTools.ttLib.scaleUpem import scale_upem
    if CJK is None: return None
    cache = Path.home() / ".cache/scimix"; cache.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    chars = set()
    for f in glob.glob(str(here / "*.py")): chars |= set(re.findall(r"[\u3000-\u9fff\uff00-\uffef℃·—]", open(f, encoding="utf-8").read()))
    key = str(abs(hash((LATIN0, CJK, "".join(sorted(chars))))))
    out = {st: cache / f"SciMix-{st}-{key}.ttf" for st in ("Regular", "Italic")}
    if not all(p.exists() for p in out.values()):
        cpath = fm.findfont(fm.FontProperties(family=CJK))
        cj = TTFont(cpath, fontNumber=0)
        opt = subset.Options(); opt.layout_features = []; opt.name_IDs = ["*"]; opt.notdef_outline = True
        ss = subset.Subsetter(opt); ss.populate(unicodes=[ord(c) for c in chars]); ss.subset(cj)
        for t in ("VORG", "vhea", "vmtx", "BASE", "GSUB", "GPOS", "GDEF", "DSIG"):
            if t in cj: del cj[t]
        if "CFF " in cj:   # CFF 轮廓 -> TrueType 二次曲线
            from fontTools.pens.cu2quPen import Cu2QuPen
            from fontTools.pens.ttGlyphPen import TTGlyphPen
            order = cj.getGlyphOrder(); gs = cj.getGlyphSet()
            glyf = newTable("glyf"); glyf.glyphOrder = order; glyf.glyphs = {}
            for n in order:
                pen = TTGlyphPen(gs); gs[n].draw(Cu2QuPen(pen, 1.0, reverse_direction=True)); glyf[n] = pen.glyph()
            cj["glyf"] = glyf; cj["loca"] = newTable("loca"); del cj["CFF "]
            mp = cj["maxp"]; mp.tableVersion = 0x00010000
            for a in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                      "maxStackElements", "maxSizeOfInstructions", "maxComponentElements", "maxPoints", "maxContours",
                      "maxCompositePoints", "maxCompositeContours", "maxComponentDepth"):
                setattr(mp, a, 1 if a == "maxZones" else 0)
            cj["head"].glyphDataFormat = 0; cj["post"].formatType = 3.0; cj.sfntVersion = "\x00\x01\x00\x00"
        for st in out:
            lp = fm.findfont(fm.FontProperties(family=LATIN0, style="italic" if st == "Italic" else "normal"))
            la = TTFont(lp, fontNumber=0)
            for t in ("DSIG", "GSUB", "GPOS", "GDEF", "kern", "hdmx", "VDMX", "LTSH", "vhea", "vmtx", "BASE", "morx", "JSTF", "MATH"):
                if t in la: del la[t]
            cjs = cj
            if cj["head"].unitsPerEm != la["head"].unitsPerEm:
                cjs = TTFont(); cj.save(cache / "tmp.ttf"); cjs = TTFont(cache / "tmp.ttf"); scale_upem(cjs, la["head"].unitsPerEm)
            la.save(cache / "a.ttf"); cjs.save(cache / "b.ttf")
            m = Merger().merge([str(cache / "a.ttf"), str(cache / "b.ttf")])
            nm = m["name"]
            for rec in list(nm.names):
                if rec.nameID in (1, 4, 6, 16, 17): nm.removeNames(nameID=rec.nameID)
            nm.setName("SciMix Serif", 1, 3, 1, 0x409); nm.setName(st, 2, 3, 1, 0x409)
            nm.setName(f"SciMix Serif {st}", 4, 3, 1, 0x409); nm.setName(f"SciMixSerif-{st}", 6, 3, 1, 0x409)
            m.save(out[st])
    for p in out.values(): fm.fontManager.addfont(str(p))
    return "SciMix Serif"


try:
    LATIN = _build_mixed_font() or LATIN0
except Exception as e:   # 合并失败时退回: 西文字体 + 中文回退 (公式内中文会缺字)
    print("混排字体构建失败, 使用回退方案:", e); LATIN = LATIN0

# Nature/NPG 风格配色 (色盲友好, 灰色用于 Baseline)
COL = {"Baseline": "#4D4D4D", "M1-调频": "#4DBBD5", "M2-热耦合": "#00A087", "M3-SP": "#3C5488",
       "M4-DRO": "#E64B35", "M5-RO": "#F39B7F", "M6-DRO+极端": "#7E6148"}
MK = {"Baseline": "o", "M1-调频": "v", "M2-热耦合": "s", "M3-SP": "D", "M4-DRO": "^", "M5-RO": "P", "M6-DRO+极端": "X"}
LAB = {"Baseline": "Baseline", "M1-调频": "M1 (GPU调节)", "M2-热耦合": "M2 (热耦合)", "M3-SP": "M3 (SP)",
       "M4-DRO": "M4 (DRO, 本文)", "M5-RO": "M5 (RO)", "M6-DRO+极端": "M6 (DRO+极端支持)"}


def _fonts(base):
    fam = [LATIN, LATIN0] + ([CJK] if CJK else [])
    plt.rcParams.update({
        "font.family": fam, "font.size": base, "axes.unicode_minus": False,
        "mathtext.fontset": "custom", "mathtext.rm": LATIN, "mathtext.it": f"{LATIN}:italic",
        "mathtext.bf": f"{LATIN}:bold", "mathtext.default": "it",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def style_a():
    _fonts(23)
    plt.rcParams.update({"axes.titlesize": 23, "axes.labelsize": 23, "xtick.labelsize": 23, "ytick.labelsize": 23,
                         "legend.fontsize": 23, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.width": 1.35, "ytick.major.width": 1.35, "xtick.major.size": 7,
                         "ytick.major.size": 7, "xtick.top": True, "ytick.right": True,
                         "lines.linewidth": 2.2, "lines.markersize": 8, "legend.frameon": False})


def frame_a(ax, right_ticks=True):
    """四边封闭: 左右 1.5 pt, 上下 1.4 pt。"""
    for s in ("left", "right"): ax.spines[s].set_linewidth(1.5); ax.spines[s].set_visible(True)
    for s in ("top", "bottom"): ax.spines[s].set_linewidth(1.4); ax.spines[s].set_visible(True)
    ax.tick_params(which="major", width=1.35, length=7, direction="in", top=True, right=right_ticks)
    ax.tick_params(which="minor", length=0)


def time_axis(ax, T=24, label="时段/$t$"):
    ax.set_xlim(0, T); ax.set_xticks(np.arange(0, T + 1, 4)); ax.set_xlabel(label)


def free_ylim(ax, *arrays, pad=0.08, n=5, lo=None):
    """按数据实际范围留白设定纵轴(不强制从 0 或整倍数起), 刻度自动选取, 使量程不呈简单比例关系。"""
    v = np.concatenate([np.ravel(a) for a in arrays]); v = v[np.isfinite(v)]
    a, b = v.min(), v.max(); r = (b - a) if b > a else max(abs(b), 1.0)
    ymin = a - pad * r if lo is None else lo
    ax.set_ylim(ymin, b + pad * r * 1.6)
    ax.yaxis.set_major_locator(MaxNLocator(n, steps=[1, 2, 2.5, 4, 5, 10]))


def style_b():
    _fonts(9)
    plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "lines.linewidth": 1.1, "lines.markersize": 4.8,
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3,
                         "ytick.major.size": 3, "legend.frameon": False, "savefig.dpi": 600})


CM = 1 / 2.54
W_SINGLE = 8.5 * CM


def save(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def stairs(ax, y, **kw):
    """时段量 (t=1..T) 以阶梯线绘制在 [0, T] 区间上。"""
    y = np.asarray(y); return ax.step(np.arange(len(y) + 1), np.r_[y, y[-1]], where="post", **kw)
