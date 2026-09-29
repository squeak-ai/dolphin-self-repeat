"""d01 repeats: what is conserved when a whistle recurs within seconds?
Local repetition (whistles within 10 s more similar than chance) survived verification. Here: decompose it.
For all within-session pairs, DTW-shape distance Ds (absolute frequency removed) and |delta f_mean|, duration ratio.
Excess of shape-matched pairs at short lag (<= 10 s) relative to long lag (60-600 s, same session), as a function of
|delta f_mean| and duration ratio:
  self-repeat (same animal, same motor program)  -> excess concentrated at delta f ~ 0, dur ratio ~ 1
  vocal matching (another animal copies the shape) -> excess also at delta f > 1 kHz
Null for the lag effect: the long-lag pairs of the same session (same repertoire, no short-term coupling).
"""
import numpy as np, pandas as pd
from d01_lib import *

df, F = load(); rng = np.random.default_rng(4); res = {}
Ds = np.load(os.path.join(OUT, 'dtw_shape.npy')); Da = np.load(os.path.join(OUT, 'dtw_abs.npy'))
# shape-match threshold: 5th percentile of all within-session pair distances (modulated whistles only)
mod = (df.cls != 'constant').values
rows = []
for s, d in df.groupby('session'):
    ix = d.index.values[mod[d.index.values]]
    if len(ix) < 20: continue
    t = df.abs_t0.values[ix]; fm = df.f_mean.values[ix] / 1000; du = df.dur.values[ix]; cl = df.cls.values[ix]
    I, J = np.triu_indices(len(ix), 1)
    lag = np.abs(t[I] - t[J])
    rows.append(pd.DataFrame(dict(session=s, i=ix[I], j=ix[J], lag=lag, ds=Ds[ix[I], ix[J]], da=Da[ix[I], ix[J]],
                                  dfm=np.abs(fm[I] - fm[J]), dr=np.maximum(du[I] / du[J], du[J] / du[I]), same_cls=(cl[I] == cl[J]))))
P = pd.concat(rows, ignore_index=True)
thr = np.percentile(P.ds, 5); res['shape_thr_p5'] = float(thr)
P['match'] = P.ds <= thr
short = P[P.lag <= 10]; long = P[(P.lag >= 60) & (P.lag <= 600)]
print(f'pairs: short-lag {len(short)}, long-lag {len(long)}; shape-match threshold (p5 of Ds) = {thr:.2f}')
res['match_rate_short'] = float(short.match.mean()); res['match_rate_long'] = float(long.match.mean())
print(f'shape-match rate: short {short.match.mean():.4f} vs long {long.match.mean():.4f}  ratio {short.match.mean()/long.match.mean():.2f}')

# (1) excess ratio by |delta f_mean| bin among shape-matched pairs: P(match & dfm in bin | short) / same at long
bins = [0, 0.5, 1, 2, 3, 5, 15]
ex = []
for lo, hi in zip(bins[:-1], bins[1:]):
    ps = ((short.match) & (short.dfm >= lo) & (short.dfm < hi)).mean(); pl = ((long.match) & (long.dfm >= lo) & (long.dfm < hi)).mean()
    # bootstrap over sessions for CI
    bs = []
    us = P.session.unique()
    for _ in range(200):
        ss = rng.choice(us, len(us))
        sh = pd.concat([short[short.session == x] for x in ss]); lg = pd.concat([long[long.session == x] for x in ss])
        a = ((sh.match) & (sh.dfm >= lo) & (sh.dfm < hi)).mean(); b = ((lg.match) & (lg.dfm >= lo) & (lg.dfm < hi)).mean()
        bs.append(a / b if b > 0 else np.nan)
    ex.append(dict(lo=lo, hi=hi, ratio=ps / pl, ci=list(np.nanpercentile(bs, [2.5, 97.5])), n_short=int(((short.match) & (short.dfm >= lo) & (short.dfm < hi)).sum())))
    print(f'  |Δf_mean| {lo:4.1f}-{hi:4.1f} kHz: short/long excess of shape-matched pairs = {ps/pl:.2f}  CI {np.round(ex[-1]["ci"],2).tolist()}  n_short={ex[-1]["n_short"]}')
res['excess_by_dfm'] = ex
# (2) same for duration ratio
ex2 = []
for lo, hi in [(1, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2), (2, 10)]:
    ps = ((short.match) & (short.dr >= lo) & (short.dr < hi)).mean(); pl = ((long.match) & (long.dr >= lo) & (long.dr < hi)).mean()
    ex2.append(dict(lo=lo, hi=hi, ratio=ps / pl)); print(f'  dur ratio {lo:.2f}-{hi:.2f}: excess {ps/pl:.2f}')
res['excess_by_dur_ratio'] = ex2
# (3) and for the absolute-frequency DTW (shape + frequency): how much of the local repetition is "exact" repetition?
thr_a = np.percentile(P.da, 5)
P['match_abs'] = P.da <= thr_a
sa = (P.lag <= 10); la = (P.lag >= 60) & (P.lag <= 600)
res['abs_match_ratio'] = float(P.match_abs[sa].mean() / P.match_abs[la].mean())
res['shape_only_match_ratio'] = float((P.match & ~P.match_abs)[sa].mean() / (P.match & ~P.match_abs)[la].mean())
print(f'exact (shape+frequency) match excess {res["abs_match_ratio"]:.2f}; shape-only (shape matches, frequency does not) excess {res["shape_only_match_ratio"]:.2f}')
# (4) lag profile of shape-matched pairs: excess vs lag
lagbins = [0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600]
prof = []
base = long.match.mean()
for lo, hi in zip(lagbins[:-1], lagbins[1:]):
    m = (P.lag >= lo) & (P.lag < hi); prof.append(dict(lo=lo, hi=hi, ratio=float(P.match[m].mean() / base), n=int(m.sum())))
res['lag_profile'] = prof
print('lag profile of shape-match rate / long-lag baseline:', [(p['lo'], round(p['ratio'], 2)) for p in prof])
# (5) same-class rate at short lag vs long: coarse-class repetition
res['same_cls_short'] = float(short.same_cls.mean()); res['same_cls_long'] = float(long.same_cls.mean())
print(f'same coarse class: short {short.same_cls.mean():.3f} vs long {long.same_cls.mean():.3f}')

# figure
f, plt = fig(11, 4)
gs = f.add_gridspec(1, 3, wspace=0.35)
ax = f.add_subplot(gs[0]); style(ax, 'a. Shape-match excess vs lag', 'lag between whistles (s), log', 'rate / long-lag baseline')
ax.plot([np.sqrt(max(p['lo'], 0.5) * p['hi']) for p in prof], [p['ratio'] for p in prof], 'o-', color=INK)
ax.axhline(1, color=MUTED, ls='--'); ax.set_xscale('log')
ax = f.add_subplot(gs[1]); style(ax, 'b. Excess by frequency shift of the repeat', '|Δ mean frequency| (kHz)', 'short-lag / long-lag excess')
xs = np.arange(len(ex)); ax.bar(xs, [e['ratio'] for e in ex], color=CLASS_COLOR['rise'], width=0.6)
ax.errorbar(xs, [e['ratio'] for e in ex], yerr=[[e['ratio'] - e['ci'][0] for e in ex], [e['ci'][1] - e['ratio'] for e in ex]], fmt='none', ecolor=INK, capsize=3, lw=1)
ax.axhline(1, color=MUTED, ls='--'); ax.set_xticks(xs); ax.set_xticklabels([f"{e['lo']}–{e['hi']}" for e in ex], fontsize=7)
ax = f.add_subplot(gs[2]); style(ax, 'c. Excess by duration ratio of the repeat', 'longer / shorter duration', 'short-lag / long-lag excess')
xs = np.arange(len(ex2)); ax.bar(xs, [e['ratio'] for e in ex2], color=CLASS_COLOR['arch'], width=0.6)
ax.axhline(1, color=MUTED, ls='--'); ax.set_xticks(xs); ax.set_xticklabels([f"{e['lo']}–{e['hi']}" for e in ex2], fontsize=7)
f.suptitle('DOLPHINFREE discovery — what a repeat conserves (shape-matched pairs, DTW-shape ≤ p5)', x=0.01, ha='left', y=1.06, fontsize=12, color=INK)
save(f, 'repeats', res)
