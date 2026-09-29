"""m03 — self-repeat / local repetition within caller (dolphin findings §3, d04 within-minute null).

Data: two-animal sessions with Fig_2 spectral features (calls2.csv). Features per call:
duration, pulses, start/end/max frequency (spectrogram bins), loudness (per-animal normalised), last-pulse length.
All z-scored globally (so individuality is retained in the data), then:

(1) Within-caller feature similarity by lag. For each call and each later call by the SAME caller in the same
    session with onset lag in window W: Euclidean distance in z-space. Statistic = mean distance in W.
    Null: permute feature vectors among the caller's own calls within the session (keeps timing, keeps caller
    identity, destroys temporal order). 200 draws. Ratio < 1 = calls closer in time are more alike.
(2) Per-feature version (|Δfeature|) for the 5–20 s window: which dimension carries the repeat?
(3) "Exact repeat" prevalence: pairs with distance ≤ p5 of all within-caller-session pair distances, at lag ≤ 30 s
    vs the same null — the marmoset analogue of the dolphin exact-pair excess / participation.
(4) Cross-caller convergence: for other-caller transitions A→B (gap ≤ 10 s), distance(B call, preceding A call)
    vs null (permute B's features among B's calls). Is the answer shaped by the question?
(5) Lag-1 autocorrelation of each feature within caller-session (Spearman), observed vs null.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from m00_lib import *

rng = np.random.default_rng(11)
c2 = pd.read_csv(os.path.join(OUT, 'calls2.csv'))
FEATS = {'dur': 'dur', 'pulses': 'f_number_of_pulses', 'f_max': 'f_maximum_frequency', 'f_start': 'f_start_frequency',
         'f_end': 'f_end_frequency', 'loud': 'f_maximum_loudness', 'pulse_len': 'f_length_of_pulse'}
X = c2[list(FEATS.values())].copy(); X.columns = list(FEATS)
X['dur'] = np.log(X.dur.clip(0.05)); X['pulse_len'] = np.log(X.pulse_len.clip(1))
ok = X.notna().all(1).values
Z = ((X - X.mean()) / X.std()).values
c2['ok'] = ok
print('calls with full features', ok.sum(), '/', len(c2))
WIN = [(5, 10), (10, 20), (20, 30), (30, 60), (60, 120), (120, 300), (300, 600), (600, 1800)]
WLAB = [f'{a}–{b}' for a, b in WIN]
NIT = 200
res = {'windows': WLAB, 'features': list(FEATS)}

# ---- build within-caller-session pair table (i<j by time)
groups = []
for (s, c), g in c2[c2.ok].groupby(['session', 'caller']):
    if len(g) < 5: continue
    groups.append((s, c, g.index.values, g.t0.values))


def pair_stats(Zg, t):
    """For one caller-session: returns per-window (sum dist, count) and per-feature |Δ| sums for 5–20 s."""
    n = len(t); i, j = np.triu_indices(n, 1); lag = t[j] - t[i]
    d = np.sqrt(((Zg[i] - Zg[j]) ** 2).sum(1))
    out = np.zeros((len(WIN), 2)); fo = np.zeros((len(FEATS), 2))
    for k, (a, b) in enumerate(WIN):
        m = (lag >= a) & (lag < b); out[k] = d[m].sum(), m.sum()
    m = (lag >= 5) & (lag < 20)
    if m.any():
        ad = np.abs(Zg[i[m]] - Zg[j[m]]); fo[:, 0] = ad.sum(0); fo[:, 1] = m.sum()
    return out, fo, d, lag


obs = np.zeros((len(WIN), 2)); fobs = np.zeros((len(FEATS), 2))
null = np.zeros((NIT, len(WIN), 2)); fnull = np.zeros((NIT, len(FEATS), 2))
per = []  # per caller-session ratio for 5–30 s
all_d, all_lag = [], []
for s, c, ix, t in groups:
    Zg = Z[ix]
    o, fo, d, lag = pair_stats(Zg, t); obs += o; fobs += fo; all_d.append(d); all_lag.append(lag)
    nn = np.zeros((NIT, len(WIN), 2)); fn = np.zeros((NIT, len(FEATS), 2))
    for it in range(NIT):
        on, fon, _, _ = pair_stats(Zg[rng.permutation(len(t))], t); nn[it] = on; fn[it] = fon
    null += nn; fnull += fn
    m = (lag >= 5) & (lag < 30)
    if m.sum() >= 10:
        ii, jj = np.triu_indices(len(t), 1)
        nv = []
        for _ in range(50):
            Zp = Zg[rng.permutation(len(t))]
            nv.append(np.sqrt(((Zp[ii[m]] - Zp[jj[m]]) ** 2).sum(1)).mean())
        per.append(dict(session=int(s), caller=c, n=len(t), npairs=int(m.sum()), obs=float(d[m].mean()), null=float(np.mean(nv))))
per = pd.DataFrame(per); per['ratio'] = per.obs / per.null
per.to_csv(os.path.join(OUT, 'm03_per_caller.csv'), index=False)

mo = obs[:, 0] / obs[:, 1]; mn = null[:, :, 0] / null[:, :, 1]
tab = pd.DataFrame(dict(window=WLAB, npairs=obs[:, 1].astype(int), obs=mo.round(4), null=mn.mean(0).round(4), sd=mn.std(0).round(4),
                        ratio=(mo / mn.mean(0)).round(4), z=((mo - mn.mean(0)) / mn.std(0)).round(1)))
print('\n(1) within-caller mean feature distance by lag (ratio < 1 = more similar than chance)'); print(tab.to_string(index=False))
res['lag_table'] = tab.to_dict('list')
w = wilcoxon(per.ratio - 1).pvalue
res['per_caller_5_30'] = dict(n=len(per), ratio_lt1=int((per.ratio < 1).sum()), median=float(per.ratio.median()), wilcoxon_p=float(w))
print(f'per caller-session (5–30 s): ratio<1 in {(per.ratio < 1).sum()}/{len(per)}, median {per.ratio.median():.3f}, p={w:.2g}')

# ---- (2) per-feature 5–20 s
fo = fobs[:, 0] / fobs[:, 1]; fn = fnull[:, :, 0] / fnull[:, :, 1]
ft = pd.DataFrame(dict(feature=list(FEATS), obs=fo.round(4), null=fn.mean(0).round(4), ratio=(fo / fn.mean(0)).round(4), z=((fo - fn.mean(0)) / fn.std(0)).round(1)))
print('\n(2) per-feature |Δ| at 5–20 s lag, obs/null'); print(ft.to_string(index=False))
res['per_feature_5_20'] = ft.to_dict('list')

# ---- (3) exact-repeat prevalence
D = np.concatenate(all_d); LG = np.concatenate(all_lag); thr = np.percentile(D, 5)
res['exact_threshold_p5'] = float(thr)
ex_obs = np.zeros(3); ex_null = np.zeros((NIT, 3)); part_obs = 0; part_null = np.zeros(NIT); ncalls = 0
LAGB = [(5, 10), (10, 30), (30, 120)]
for s, c, ix, t in groups:
    Zg = Z[ix]; n = len(t); i, j = np.triu_indices(n, 1); lag = t[j] - t[i]
    def cnt(Zx):
        d = np.sqrt(((Zx[i] - Zx[j]) ** 2).sum(1)); e = d <= thr
        out = np.array([(e & (lag >= a) & (lag < b)).sum() for a, b in LAGB])
        pm = e & (lag < 30); part = len(np.unique(np.concatenate([i[pm], j[pm]])))
        return out, part
    o, p = cnt(Zg); ex_obs += o; part_obs += p; ncalls += n
    for it in range(NIT):
        o, p = cnt(Zg[rng.permutation(n)]); ex_null[it] += o; part_null[it] += p
et = pd.DataFrame(dict(lag=[f'{a}–{b}' for a, b in LAGB], obs=ex_obs.astype(int), null=ex_null.mean(0).round(1), sd=ex_null.std(0).round(1),
                       ratio=(ex_obs / ex_null.mean(0)).round(3), z=((ex_obs - ex_null.mean(0)) / ex_null.std(0)).round(1)))
print('\n(3) exact-repeat pairs (distance ≤ p5) by lag'); print(et.to_string(index=False))
print(f'participation (calls in an exact pair ≤ 30 s): obs {part_obs} ({100*part_obs/ncalls:.1f}%) vs null {part_null.mean():.0f} ({100*part_null.mean()/ncalls:.1f}%) → excess {100*(part_obs-part_null.mean())/ncalls:.2f} pct points')
res['exact'] = et.to_dict('list'); res['participation'] = dict(obs=int(part_obs), null=float(part_null.mean()), null_sd=float(part_null.std()), ncalls=int(ncalls),
                                                              excess_pct=float(100 * (part_obs - part_null.mean()) / ncalls))

# ---- (4) cross-caller convergence: B's call vs the A call it answers (gap ≤ 10 s)
conv_obs = {f: 0.0 for f in FEATS}; conv_obs['all'] = 0.0; conv_null = {k: np.zeros(NIT) for k in conv_obs}; npair = 0
for s, g in c2[c2.ok].groupby('session'):
    g = g.sort_values('t0'); ix = g.index.values; t0, t1, c = g.t0.values, g.t1.values, g.caller.values; Zg = Z[ix]
    prev = np.arange(len(g)) - 1
    m = np.zeros(len(g), bool); m[1:] = (c[1:] != c[:-1]) & ((t0[1:] - t1[:-1]) <= 10) & ((t0[1:] - t1[:-1]) >= 0)
    if m.sum() < 3: continue
    bi = np.where(m)[0]; ai = bi - 1
    def stats(Zx):
        dd = Zx[bi] - Zx[ai]; return dict(all=np.sqrt((dd ** 2).sum(1)).sum(), **{f: np.abs(dd[:, k]).sum() for k, f in enumerate(FEATS)})
    o = stats(Zg); npair += m.sum()
    for k in conv_obs: conv_obs[k] += o[k]
    callers = np.unique(c)
    for it in range(NIT):
        Zp = Zg.copy()
        for cc in callers:
            mm = c == cc; Zp[mm] = Zg[mm][rng.permutation(mm.sum())]
        o = stats(Zp)
        for k in conv_obs: conv_null[k][it] += o[k]
ct = pd.DataFrame([dict(feature=k, obs=conv_obs[k] / npair, null=conv_null[k].mean() / npair, ratio=conv_obs[k] / conv_null[k].mean(),
                        z=(conv_obs[k] - conv_null[k].mean()) / conv_null[k].std()) for k in conv_obs]).round(4)
print(f'\n(4) cross-caller convergence, A→B transitions with gap ≤ 10 s (n={npair}): distance(B, A) obs/null (ratio<1 = answer resembles question)')
print(ct.to_string(index=False)); res['convergence'] = ct.to_dict('list'); res['convergence_n'] = int(npair)

# ---- (5) lag-1 autocorrelation per feature within caller-session
ac = {f: [] for f in FEATS}; acn = {f: [] for f in FEATS}
for s, c, ix, t in groups:
    if len(ix) < 10: continue
    for k, f in enumerate(FEATS):
        x = Z[ix, k]; ac[f].append(spearmanr(x[:-1], x[1:])[0])
        acn[f].append(np.mean([spearmanr(*(lambda y: (y[:-1], y[1:]))(rng.permutation(x)))[0] for _ in range(20)]))
at = pd.DataFrame([dict(feature=f, rho_obs=np.nanmean(ac[f]), rho_null=np.nanmean(acn[f]), n=len(ac[f]),
                        gt0=int((np.array(ac[f]) > 0).sum()), wilcoxon_p=wilcoxon(np.array(ac[f]) - np.array(acn[f])).pvalue) for f in FEATS]).round(4)
print('\n(5) lag-1 Spearman autocorrelation within caller-session (consecutive own calls)'); print(at.to_string(index=False))
res['lag1_autocorr'] = at.to_dict('list')
dump('m03_selfrepeat', res)
