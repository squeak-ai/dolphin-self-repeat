"""Shared publication style for all article figures (dolphin d04, cross-species m06, Null B m07, spectrograms).

One palette, one type scale, one save routine.
  Species colours are fixed across every figure: dolphin = blue, marmoset = copper. Nulls = neutral greys.
  Type scale at the nominal full-width figure (10 in): 12 pt axis labels / titles, 10 pt ticks and legends,
  9.5 pt annotations, 14 pt bold panel letters, 13 pt figure headline.
  Output: PNG at 300 dpi + vector PDF (fonts embedded as TrueType).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- palette (dataviz reference, light surface) ----
INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#9a9891'; GRID = '#e6e5e1'; SURF = '#fcfcfb'
BLUE = '#2a78d6'; COPPER = '#C4883A'
DOLPHIN = BLUE; MARMOSET = COPPER
DOLPHIN_LIGHT = '#a9c8ee'; MARMOSET_LIGHT = '#e8c9a8'
DOLPHIN_DARK = '#1a5499'                                  # secondary dolphin series (e.g. transposed triples)
BLUES = ['#a9c8ee', '#6ba3e0', '#2a78d6', '#1a5499', '#0f3466']   # light -> dark, repeated renditions
# Legacy — used by spectrogram colormap (d06), not species-coded
ORANGE = '#eb6834'
ORANGES = BLUES  # backward compat alias
G1 = '#c3c2b7'; G2 = '#8a8984'; G3 = '#52514e'           # null greys, light -> dark
BAND_NEUTRAL = '#f3f2ee'; BAND_WARM = '#fbeee6'          # background spans

# ---- type scale ----
FS_LABEL = 12; FS_TICK = 10; FS_LEGEND = 10; FS_ANNO = 9.5; FS_SMALL = 9; FS_TITLE = 12; FS_HEAD = 13; FS_PANEL = 14
FULL_W = 10.0

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': FS_TICK,
    'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_LABEL, 'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEGEND, 'legend.title_fontsize': FS_SMALL, 'legend.frameon': False,
    'savefig.facecolor': SURF, 'figure.facecolor': SURF, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FS_TICK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, color=INK, fontsize=FS_TITLE, loc='left', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=INK2, fontsize=FS_LABEL)
    if ylabel: ax.set_ylabel(ylabel, color=INK2, fontsize=FS_LABEL)


def panel_label(ax, s, x=-0.08, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', color=INK, va='bottom', ha='right')


def headline(f, text, y=1.02, sub=None, sub_y=None):
    f.suptitle(text, x=0.01, y=y, ha='left', fontsize=FS_HEAD, color=INK)
    if sub:
        f.text(0.01, sub_y if sub_y is not None else y - 0.06, sub, fontsize=FS_SMALL, color=INK2, ha='left', va='top')


def save_both(f, outdir, name):
    for ext, kw in [('png', dict(dpi=300)), ('pdf', {})]:
        p = os.path.join(outdir, f'{name}.{ext}')
        f.savefig(p, bbox_inches='tight', facecolor=SURF, **kw)
        print('saved', p)
    plt.close(f)
