"""Shared loader for dolphin whistle analysis.

Corrected methodology: bandwidth gate (< 1 kHz → 'constant') before shape classes;
spectral k=4 for the modulated whistles.

Everything is aligned by row order of whistles_obs.csv == features.pkl arrays == v02_spectral_labels.csv.

Data source: Lehnhoff et al. 2025, Zenodo DOI: 10.5281/zenodo.14637674
"""
import os, json, pickle
import numpy as np, pandas as pd

# Paths — override via environment variables or edit here.
# DATA_DIR should point to the directory containing whistles_obs.csv, features.pkl, etc.
OUT = os.environ.get('DOLPHIN_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'dolphin'))
FIG = os.environ.get('DOLPHIN_FIG_DIR', os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIG, exist_ok=True)
SPEC4_NAMES = {0: 'arch', 1: 'rise', 2: 'fall', 3: 'U'}
CLASS_ORDER = ['rise', 'fall', 'arch', 'U', 'constant']
# dataviz reference palette (light mode), fixed order per class
CLASS_COLOR = {'rise': '#2a78d6', 'fall': '#eb6834', 'arch': '#1baf7a', 'U': '#eda100', 'constant': '#8a8984'}
INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#9a9891'; GRID = '#e6e5e1'; SURF = '#fcfcfb'
BW_GATE = 1000.0


def load():
    df = pd.read_csv(os.path.join(OUT, 'whistles_obs.csv'))
    F = pickle.load(open(os.path.join(OUT, 'features.pkl'), 'rb'))
    lab = pd.read_csv(os.path.join(OUT, 'v02_spectral_labels.csv'))
    assert (df.idx.values == lab.idx.values).all()
    df['spec4'] = lab.spec4.values
    df['cls'] = np.where(df.bw < BW_GATE, 'constant', df.spec4.map(SPEC4_NAMES))
    df['cls'] = pd.Categorical(df['cls'], CLASS_ORDER)
    # encounter id (ID_group within year) and session already present
    df['enc'] = df.year.astype(str) + '_' + df.ID_group.astype(str)
    df['playback'] = df.playback.fillna('silent')
    return df, F


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, color=INK, fontsize=10, loc='left', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=INK2, fontsize=8)


def fig(w=10, h=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(w, h), facecolor=SURF)
    return f, plt


def save(f, name, res=None):
    p = os.path.join(FIG, f'discovery_{name}.png')
    f.savefig(p, dpi=150, facecolor=SURF, bbox_inches='tight')
    print('saved', p)
    if res is not None:
        with open(os.path.join(OUT, f'd01_{name}.json'), 'w') as fh:
            json.dump(res, fh, indent=1, default=float)


def perm_p(obs, null, side='two'):
    null = np.asarray(null)
    if side == 'greater':
        return (1 + (null >= obs).sum()) / (1 + len(null))
    if side == 'less':
        return (1 + (null <= obs).sum()) / (1 + len(null))
    return (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r[o] = np.minimum(q, 1); return r
