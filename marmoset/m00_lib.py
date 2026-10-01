"""Marmoset replication — shared loader and plotting style.

Data: Grijseels, Fairbank & Miller 2024, Dryad doi:10.5061/dryad.9ghx3ffpx.
Each session .mat holds `call_data` (1 x n_animals cell; each cell an (n_calls, 2) array of
(start_frame, end_frame)) and `all_pulses` (pulses per call). Frames at 41.3302 fps.

Version choice (see m01_load.py output): two_monkey -> call_data_v2 (fallback v1);
one_monkey -> highest of v3 > v2 > v1_reextract > v1; three_monkey -> v2cleaned.
"""
import os, re, glob, json
import numpy as np, pandas as pd
import scipy.io as sio

# Paths — override via environment variables or edit here.
# MARMOSET_RAW_DIR should point to the extracted Dryad dataset root (vocal_turn_taking_data/).
RAW = os.environ.get('MARMOSET_RAW_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'marmoset', 'vocal_turn_taking_data'))
OUT = os.environ.get('MARMOSET_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'marmoset'))
FIG = os.environ.get('MARMOSET_FIG_DIR', os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
_FPS_FILE = os.path.join(RAW, 'metadata/fps.txt')
if not os.path.exists(_FPS_FILE):
    raise FileNotFoundError(f'Marmoset dataset not found at {os.path.abspath(RAW)} (expected metadata/fps.txt). '
                            'Download the Dryad deposit (doi:10.5061/dryad.9ghx3ffpx) and extract it to data/marmoset/vocal_turn_taking_data/, '
                            'or set MARMOSET_RAW_DIR.')
FPS = float(open(_FPS_FILE).read().strip())

# dataviz reference palette (light surface) — same as d01_lib
INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#9a9891'; GRID = '#e6e5e1'; SURF = '#fcfcfb'
BLUE = '#2a78d6'; ORANGE = '#eb6834'; GREEN = '#1baf7a'; AMBER = '#eda100'; GREY = '#8a8984'
COPPER = '#C4883A'
DOLPHIN = BLUE; MARMOSET = COPPER  # species palette (matches figstyle.py)

_VER_RANK = {'v3': 4, 'v2cleaned': 4, 'v2': 3, 'v1_reextract': 2, 'v1': 1}


def _version(path):
    m = re.search(r'call_data_(v\d\w*)\.mat$', os.path.basename(path))
    return m.group(1) if m else None


def _session_files():
    """Yield (context, session_key, chosen_path, all_versions)."""
    for ctx, n in [('one_monkey', 1), ('two_monkey', 2), ('three_monkey', 3)]:
        files = glob.glob(os.path.join(RAW, ctx, '**', '*.mat'), recursive=True)
        groups = {}
        for f in files:
            if ctx == 'two_monkey':
                key = os.path.basename(os.path.dirname(f))
            else:
                key = os.path.basename(f).split('_call_data_')[0]
            groups.setdefault(key, []).append(f)
        for key, fs in sorted(groups.items()):
            best = max(fs, key=lambda p: _VER_RANK.get(_version(p), 0))
            yield ctx, n, key, best, sorted(_version(p) for p in fs)


def _names_from_key(key, n):
    # e.g. 'cain_locke_20220325', 'athena_20230213', 'cassi_loretta_leo_20230227ttl'
    parts = key.split('_')
    date = re.sub(r'\D', '', parts[-1])[:8]
    names = [p.lower() for p in parts[:-1]]
    if len(names) != n:
        names = (names + ['?'] * n)[:n]
    return names, date


def load_calls(cache=True):
    """One row per call. Columns: ctx (1/2/3), session (int), key, date, caller (name),
    caller_idx (position in cell array), t0, t1 (s), dur (s), pulses, version."""
    cp = os.path.join(OUT, 'calls.csv')
    if cache and os.path.exists(cp):
        return pd.read_csv(cp)
    rows = []; sid = 0
    for ctx, n, key, path, vers in _session_files():
        m = sio.loadmat(path)
        cd = m['call_data']; ap = m.get('all_pulses')
        names, date = _names_from_key(key, n)
        sid += 1
        ncell = cd.shape[1] if cd.ndim == 2 else len(cd)
        for i in range(ncell):
            arr = np.asarray(cd[0, i], dtype=float)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, 2)
            pulses = None
            if ap is not None:
                pa = np.asarray(ap[0, i]).reshape(-1)
                if len(pa) == len(arr):
                    pulses = pa.astype(float)
            for j, (a, b) in enumerate(arr):
                rows.append(dict(ctx=n, session=sid, key=key, date=date, caller=names[i] if i < len(names) else f'a{i}',
                                 caller_idx=i, t0=a / FPS, t1=b / FPS, dur=(b - a) / FPS,
                                 pulses=(pulses[j] if pulses is not None else np.nan), version=_version(path)))
    df = pd.DataFrame(rows)
    df = df.sort_values(['session', 't0']).reset_index(drop=True)
    df['idx'] = np.arange(len(df))
    if cache:
        df.to_csv(cp, index=False)
    return df


def load_features():
    """Fig_2_features.csv — per-call spectral features for two-monkey sessions."""
    f = pd.read_csv(os.path.join(RAW, 'metadata/Fig_2_features.csv'), encoding='utf-8-sig')
    f.columns = [c.strip() for c in f.columns]
    return f


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
    p = os.path.join(FIG, f'marmoset_{name}.png')
    f.savefig(p, dpi=150, facecolor=SURF, bbox_inches='tight')
    print('saved', p)
    if res is not None:
        with open(os.path.join(OUT, f'm_{name}.json'), 'w') as fh:
            json.dump(res, fh, indent=1, default=float)


def dump(name, res):
    with open(os.path.join(OUT, f'{name}.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print('wrote', name + '.json')


def perm_p(obs, null, side='two'):
    null = np.asarray(null)
    if side == 'greater':
        return (1 + (null >= obs).sum()) / (1 + len(null))
    if side == 'less':
        return (1 + (null <= obs).sum()) / (1 + len(null))
    return (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null))


def burstiness(iois):
    iois = np.asarray(iois, float)
    if len(iois) < 3: return np.nan
    m, s = iois.mean(), iois.std()
    return (s - m) / (s + m)


def fano(onsets, T, bins):
    """Fano factor of counts at each bin width (s) over [0, T]."""
    out = {}
    for w in bins:
        nb = int(np.floor(T / w))
        if nb < 5: out[w] = np.nan; continue
        c, _ = np.histogram(onsets, bins=nb, range=(0, nb * w))
        out[w] = c.var() / c.mean() if c.mean() > 0 else np.nan
    return out
