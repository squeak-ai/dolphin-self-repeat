"""m03b — the marmoset analogue of the dolphin within-minute null.

m03's session-wide permutation confounds short-lag similarity with slow drift (ratio < 1 out to 300 s, > 1 at
600–1800 s). Here the null permutes feature vectors only among a caller's calls inside non-overlapping time
blocks (60 s, 120 s, 300 s). Whatever similarity survives the 60-s-block null is faster than minute-scale drift —
the same logic as d04's within-minute null for dolphins. Also: feature autocorrelation by call-lag (1…12 own calls),
and everything with and without loudness (the feature most exposed to position/orientation drift).
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from m00_lib import *

rng = np.random.default_rng(23)
c2 = pd.read_csv(os.path.join(OUT, 'calls2.csv'))
FEATS = {'dur': 'dur', 'pulses': 'f_number_of_pulses', 'f_max': 'f_maximum_frequency', 'f_start': 'f_start_frequency',
         'f_end': 'f_end_frequency', 'loud': 'f_maximum_loudness', 'pulse_len': 'f_length_of_pulse'}
X = c2[list(FEATS.values())].copy(); X.columns = list(FEATS)
X['dur'] = np.log(X.dur.clip(0.05)); X['pulse_len'] = np.log(X.pulse_len.clip(1))
ok = X.notna().all(1).values
Z = ((X - X.mean()) / X.std()).values
NOLOUD = [k for k, f in enumerate(FEATS) if f != 'loud']
WIN = [(5, 10), (10, 20), (20, 30), (30, 60), (60, 120), (120, 300)]
WLAB = [f'{a}–{b}' for a, b in WIN]
BLOCKS = [60, 120, 300, None]
NIT = 200
res = {'windows': WLAB}
groups = [(s, c, g.index.values, g.t0.values) for (s, c), g in c2[ok].groupby(['session', 'caller']) if len(g) >= 5]


def block_perm(t, B):
    p = np.arange(len(t))
    if B is None: return rng.permutation(p)
    b = (t // B).astype(int)
    for u in np.unique(b):
        m = np.where(b == u)[0]
        if len(m) > 1: p[m] = m[rng.permutation(len(m))]
    return p


def wsum(Zg, t, cols):
    n = len(t); i, j = np.triu_indices(n, 1); lag = t[j] - t[i]
    d = np.sqrt(((Zg[i][:, cols] - Zg[j][:, cols]) ** 2).sum(1))
    return np.array([[d[(lag >= a) & (lag < b)].sum(), ((lag >= a) & (lag < b)).sum()] for a, b in WIN])


for tag, cols in [('all7', list(range(len(FEATS)))), ('noloud', NOLOUD)]:
    print(f'\n=== feature set: {tag}')
    res[tag] = {}
    for B in BLOCKS:
        obs = np.zeros((len(WIN), 2)); null = np.zeros((NIT, len(WIN), 2)); per = []
        for s, c, ix, t in groups:
            Zg = Z[ix]; o = wsum(Zg, t, cols); obs += o
            nn = np.zeros((NIT, len(WIN), 2))
            for it in range(NIT):
                nn[it] = wsum(Zg[block_perm(t, B)], t, cols)
            null += nn
            if o[0:2, 1].sum() >= 10:  # 5–20 s pairs
                per.append((o[0:2, 0].sum() / o[0:2, 1].sum()) / (nn[:, 0:2, 0].sum(1) / nn[:, 0:2, 1].sum(1)).mean())
        mo = obs[:, 0] / obs[:, 1]; mn = null[:, :, 0] / null[:, :, 1]
        ratio = mo / mn.mean(0); z = (mo - mn.mean(0)) / mn.std(0)
        per = np.array(per); w = wilcoxon(per - 1).pvalue if len(per) > 5 else np.nan
        key = f'block{B}' if B else 'session'
        res[tag][key] = dict(ratio=ratio.round(4).tolist(), z=z.round(1).tolist(), npairs=obs[:, 1].astype(int).tolist(),
                             per_caller_n=int(len(per)), per_caller_lt1=int((per < 1).sum()), per_caller_median=float(np.median(per)), wilcoxon_p=float(w))
        print(f'  null block={str(B):>5}: ratio ' + ' '.join(f'{WLAB[k]}={ratio[k]:.3f}(z{z[k]:+.0f})' for k in range(len(WIN)) if not np.isnan(ratio[k])),
              f'| per caller-session 5–20 s: {(per < 1).sum()}/{len(per)} <1, median {np.median(per):.3f}, p={w:.1e}')

# ---- autocorrelation by call lag (own consecutive calls), per feature, vs 60-s-block null
LAGS = list(range(1, 13))
res['acf'] = {}
print('\n=== Spearman autocorrelation by own-call lag (mean over caller-sessions with ≥ 20 calls); null = 60-s block permutation')
for k, f in enumerate(FEATS):
    ro = np.full((0, len(LAGS)), np.nan); rn = []
    rows_o, rows_n = [], []
    for s, c, ix, t in groups:
        if len(ix) < 20: continue
        x = Z[ix, k]
        rows_o.append([spearmanr(x[:-l], x[l:])[0] for l in LAGS])
        rows_n.append(np.mean([[spearmanr(*(lambda y: (y[:-l], y[l:]))(x[block_perm(t, 60)]))[0] for l in LAGS] for _ in range(20)], 0))
    ro, rn = np.nanmean(rows_o, 0), np.nanmean(rows_n, 0)
    res['acf'][f] = dict(obs=ro.round(3).tolist(), null60=rn.round(3).tolist(), n=len(rows_o))
    print(f'  {f:>9}: obs  ' + ' '.join(f'{v:+.2f}' for v in ro) + f'   | null60 lag1 {rn[0]:+.2f} lag5 {rn[4]:+.2f} lag12 {rn[11]:+.2f}')
dump('m03b_localnull', res)
