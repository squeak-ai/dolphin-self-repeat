"""d02: decompose the d01 self-repeat finding. Is the depletion of shape-matched-but-pitch-shifted pairs at short lag
(a) an interaction (shape match is LESS likely at short lag given a pitch shift), or
(b) a marginal effect (short-lag pairs are simply closer in frequency/duration, shape-independent)?

d01_repeats reported P(match & dfm-bin | short) / P(match & dfm-bin | long): a JOINT rate. That ratio can go below 1
for a bin even if shape match is independent of lag, whenever the marginal P(dfm-bin | short) is below P(dfm-bin | long).
Here: marginal ratios, conditional ratios, overlap vs gap split, lag-resolved exact-match excess, and a
joint (dfm, dur) conditional table. Same pair construction as d01_repeats.py.
"""
import numpy as np, pandas as pd, json
from d01_lib import *

df, F = load(); rng = np.random.default_rng(7); res = {}
Ds = np.load(os.path.join(OUT, 'dtw_shape.npy')); Da = np.load(os.path.join(OUT, 'dtw_abs.npy'))
mod = (df.cls != 'constant').values
rows = []
for s, d in df.groupby('session'):
    ix = d.index.values[mod[d.index.values]]
    if len(ix) < 20: continue
    t0 = df.abs_t0.values[ix]; t1 = df.abs_t1.values[ix]; fm = df.f_mean.values[ix] / 1000; du = df.dur.values[ix]; cl = df.cls.values[ix]
    I, J = np.triu_indices(len(ix), 1)
    # order so that i is the earlier whistle
    first = np.where(t0[I] <= t0[J], I, J); second = np.where(t0[I] <= t0[J], J, I)
    lag = t0[second] - t0[first]
    overlap = t0[second] < t1[first]          # second starts before first ends -> two voices (or biphonation)
    gap = t0[second] - t1[first]             # negative when overlapping
    rows.append(pd.DataFrame(dict(session=s, i=ix[first], j=ix[second], lag=lag, gap=gap, overlap=overlap,
                                  ds=Ds[ix[first], ix[second]], da=Da[ix[first], ix[second]],
                                  dfm=np.abs(fm[first] - fm[second]), dr=np.maximum(du[first] / du[second], du[second] / du[first]),
                                  same_cls=(cl[first] == cl[second]))))
P = pd.concat(rows, ignore_index=True)
thr = np.percentile(P.ds, 5); thr_a = np.percentile(P.da, 5)
P['match'] = P.ds <= thr; P['match_abs'] = P.da <= thr_a
short = P[P.lag <= 10]; long = P[(P.lag >= 60) & (P.lag <= 600)]
res['n_short'] = int(len(short)); res['n_long'] = int(len(long)); res['thr_shape'] = float(thr); res['thr_abs'] = float(thr_a)
print(f'short-lag pairs {len(short)}, long-lag {len(long)}; overlapping among short: {short.overlap.mean():.3f}')

sessions = P.session.unique()
def boot(fn, n=300):
    out = []
    for _ in range(n):
        ss = rng.choice(sessions, len(sessions))
        sh = pd.concat([short[short.session == x] for x in ss]); lg = pd.concat([long[long.session == x] for x in ss])
        out.append(fn(sh, lg))
    return np.nanpercentile(out, [2.5, 97.5]).tolist()

# ---------- 1. marginal vs conditional by |delta f_mean| ----------
bins = [0, 0.5, 1, 2, 3, 5, 15]
tab = []
print('\n|Δf_mean| bin: joint ratio (d01) | marginal P(bin|short)/P(bin|long) | conditional P(match|bin,short)/P(match|bin,long)')
for lo, hi in zip(bins[:-1], bins[1:]):
    inb = lambda x: (x.dfm >= lo) & (x.dfm < hi)
    joint = (short.match & inb(short)).mean() / (long.match & inb(long)).mean()
    marg = inb(short).mean() / inb(long).mean()
    cond = short.match[inb(short)].mean() / long.match[inb(long)].mean()
    ci_m = boot(lambda a, b: inb(a).mean() / inb(b).mean())
    ci_c = boot(lambda a, b: a.match[inb(a)].mean() / b.match[inb(b)].mean())
    tab.append(dict(lo=lo, hi=hi, joint=joint, marginal=marg, marginal_ci=ci_m, conditional=cond, conditional_ci=ci_c,
                    n_short_pairs=int(inb(short).sum()), n_short_matches=int((short.match & inb(short)).sum()),
                    p_bin_short=float(inb(short).mean()), p_bin_long=float(inb(long).mean())))
    print(f'  {lo:4.1f}-{hi:4.1f}: joint {joint:.2f} | marginal {marg:.2f} [{ci_m[0]:.2f},{ci_m[1]:.2f}] | conditional {cond:.2f} [{ci_c[0]:.2f},{ci_c[1]:.2f}]  n={inb(short).sum()}')
res['by_dfm'] = tab

# ---------- 2. same for duration ratio ----------
dbins = [(1, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2), (2, 10)]
tab2 = []
print('\nduration ratio bin: joint | marginal | conditional')
for lo, hi in dbins:
    inb = lambda x: (x.dr >= lo) & (x.dr < hi)
    joint = (short.match & inb(short)).mean() / (long.match & inb(long)).mean()
    marg = inb(short).mean() / inb(long).mean()
    cond = short.match[inb(short)].mean() / long.match[inb(long)].mean()
    ci_c = boot(lambda a, b: a.match[inb(a)].mean() / b.match[inb(b)].mean())
    tab2.append(dict(lo=lo, hi=hi, joint=joint, marginal=marg, conditional=cond, conditional_ci=ci_c, n_short_pairs=int(inb(short).sum())))
    print(f'  {lo:.2f}-{hi:.2f}: joint {joint:.2f} | marginal {marg:.2f} | conditional {cond:.2f} [{ci_c[0]:.2f},{ci_c[1]:.2f}]')
res['by_dur'] = tab2

# ---------- 3. joint conditional: pitch AND duration both close vs not ----------
close_f = lambda x: x.dfm < 0.5; close_d = lambda x: x.dr < 1.1
cells = {}
for name, sel in [('f close & d close', lambda x: close_f(x) & close_d(x)), ('f close & d far', lambda x: close_f(x) & ~close_d(x)),
                  ('f far(>1) & d close', lambda x: (x.dfm >= 1) & close_d(x)), ('f far(>1) & d far', lambda x: (x.dfm >= 1) & ~close_d(x))]:
    cond = short.match[sel(short)].mean() / long.match[sel(long)].mean()
    ci = boot(lambda a, b: a.match[sel(a)].mean() / b.match[sel(b)].mean())
    cells[name] = dict(conditional=float(cond), ci=ci, n_short=int(sel(short).sum()), n_short_match=int((short.match & sel(short)).sum()))
    print(f'  {name}: conditional match excess {cond:.2f} [{ci[0]:.2f},{ci[1]:.2f}]  n_short={sel(short).sum()}')
res['joint_cells'] = cells

# ---------- 4. overlapping vs non-overlapping short-lag pairs ----------
print('\noverlap split (short lag <= 10 s):')
ov = {}
for name, sel in [('overlapping', lambda x: x.overlap), ('gap 0-1 s', lambda x: (~x.overlap) & (x.gap < 1)), ('gap 1-3 s', lambda x: (~x.overlap) & (x.gap >= 1) & (x.gap < 3)), ('gap 3-10 s', lambda x: (~x.overlap) & (x.gap >= 3))]:
    sh = short[sel(short)]
    e_abs = sh.match_abs.mean() / long.match_abs.mean(); e_shape = sh.match.mean() / long.match.mean()
    e_shape_only = (sh.match & ~sh.match_abs).mean() / (long.match & ~long.match_abs).mean()
    e_cond_close = sh.match[close_f(sh)].mean() / long.match[close_f(long)].mean()
    e_cond_far = sh.match[sh.dfm >= 1].mean() / long.match[long.dfm >= 1].mean()
    marg_close = close_f(sh).mean() / close_f(long).mean()
    ci = boot(lambda a, b: a[sel(a)].match_abs.mean() / b.match_abs.mean())
    ov[name] = dict(n=int(len(sh)), exact_excess=float(e_abs), exact_ci=ci, shape_excess=float(e_shape), shape_only_excess=float(e_shape_only),
                    cond_close_f=float(e_cond_close), cond_far_f=float(e_cond_far), marginal_close_f=float(marg_close))
    print(f'  {name:12s} n={len(sh):6d}: exact {e_abs:.2f} [{ci[0]:.2f},{ci[1]:.2f}] | shape {e_shape:.2f} | shape-only {e_shape_only:.2f} | cond(f close) {e_cond_close:.2f} | cond(f far) {e_cond_far:.2f} | marginal P(f close) {marg_close:.2f}')
res['overlap_split'] = ov

# ---------- 5. lag-resolved: exact match, shape-only match, marginal f-closeness ----------
lagbins = [0, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600]
prof = []
b_abs = long.match_abs.mean(); b_so = (long.match & ~long.match_abs).mean(); b_cf = close_f(long).mean(); b_m = long.match.mean()
b_cond_close = long.match[close_f(long)].mean(); b_cond_far = long.match[long.dfm >= 1].mean()
print('\nlag profile: exact | shape-only | shape | P(f close) | cond(f close) | cond(f far)')
for lo, hi in zip(lagbins[:-1], lagbins[1:]):
    m = P[(P.lag >= lo) & (P.lag < hi)]
    r = dict(lo=lo, hi=hi, n=int(len(m)), exact=float(m.match_abs.mean() / b_abs), shape_only=float((m.match & ~m.match_abs).mean() / b_so),
             shape=float(m.match.mean() / b_m), p_close_f=float(close_f(m).mean() / b_cf),
             cond_close_f=float(m.match[close_f(m)].mean() / b_cond_close), cond_far_f=float(m.match[m.dfm >= 1].mean() / b_cond_far),
             n_exact=int(m.match_abs.sum()))
    prof.append(r)
    print(f'  {lo:5.1f}-{hi:5.1f} n={len(m):7d}: exact {r["exact"]:.2f} (n={r["n_exact"]}) | shape-only {r["shape_only"]:.2f} | shape {r["shape"]:.2f} | P(f close) {r["p_close_f"]:.2f} | cond close {r["cond_close_f"]:.2f} | cond far {r["cond_far_f"]:.2f}')
res['lag_profile'] = prof

# ---------- 6. pure frequency autocorrelation, shape-free: |Δf_mean| median vs lag; same for duration ----------
print('\nmedian |Δf_mean| (kHz) and median dur ratio by lag:')
fa = []
for lo, hi in zip(lagbins[:-1], lagbins[1:]):
    m = P[(P.lag >= lo) & (P.lag < hi)]
    fa.append(dict(lo=lo, hi=hi, med_dfm=float(m.dfm.median()), med_dr=float(m.dr.median()), same_cls=float(m.same_cls.mean())))
    print(f'  {lo:5.1f}-{hi:5.1f}: |Δf| {m.dfm.median():.2f}  dur ratio {m.dr.median():.2f}  same class {m.same_cls.mean():.3f}')
res['freq_autocorr'] = fa

# ---------- 7. per-session replication of the conditional close-f excess and the conditional far-f ratio ----------
per = []
for s in sessions:
    sh = short[short.session == s]; lg = long[long.session == s]
    if len(sh) < 200 or len(lg) < 500: continue
    a = sh.match[close_f(sh)].mean(); b = lg.match[close_f(lg)].mean(); c = sh.match[sh.dfm >= 1].mean(); d = lg.match[lg.dfm >= 1].mean()
    per.append(dict(session=int(s), n_short=int(len(sh)), cond_close=float(a / b) if b > 0 else np.nan, cond_far=float(c / d) if d > 0 else np.nan))
per = pd.DataFrame(per)
from scipy.stats import wilcoxon
pc = per.cond_close.dropna(); pf = per.cond_far.dropna()
print(f'\nper-session (n={len(per)}): cond close-f excess >1 in {(pc > 1).sum()}/{len(pc)} (Wilcoxon p={wilcoxon(np.log(pc)).pvalue:.3g}, median {pc.median():.2f}); '
      f'cond far-f <1 in {(pf < 1).sum()}/{len(pf)} (p={wilcoxon(np.log(pf)).pvalue:.3g}, median {pf.median():.2f})')
res['per_session'] = per.to_dict(orient='records')
res['per_session_summary'] = dict(n=int(len(per)), close_gt1=int((pc > 1).sum()), close_p=float(wilcoxon(np.log(pc)).pvalue), close_median=float(pc.median()),
                                  far_lt1=int((pf < 1).sum()), far_p=float(wilcoxon(np.log(pf)).pvalue), far_median=float(pf.median()))

# ---------- 8. how exact is exact? distribution of dfm and dr among short-lag exact matches vs long-lag exact matches ----------
se = short[short.match_abs]; le = long[long.match_abs]
res['exact_pairs'] = dict(n_short=int(len(se)), n_long=int(len(le)), med_dfm_short=float(se.dfm.median()), med_dfm_long=float(le.dfm.median()),
                          med_dr_short=float(se.dr.median()), med_dr_long=float(le.dr.median()), med_ds_short=float(se.ds.median()), med_ds_long=float(le.ds.median()))
print(f'\nexact matches: short n={len(se)} |Δf| med {se.dfm.median():.2f} dr {se.dr.median():.3f} ds {se.ds.median():.2f}; long n={len(le)} |Δf| {le.dfm.median():.2f} dr {le.dr.median():.3f} ds {le.ds.median():.2f}')

# ---------- 9. the compositional check: additive self-repeat model predicts total shape-match ratio ----------
tot = short.match.mean() / long.match.mean()
excess_pairs = (short.match & close_f(short)).sum() - (short.match & close_f(short)).sum() / tab[0]['joint']
print(f'\ntotal shape-match ratio short/long = {tot:.3f}; additive excess in f<0.5 bin ≈ {excess_pairs:.0f} pairs of {short.match.sum()} short-lag matches')
res['total_shape_ratio'] = float(tot)

with open(os.path.join(OUT, 'd02_repeats_decomp.json'), 'w') as fh:
    json.dump(res, fh, indent=1, default=float)
print('saved d02_repeats_decomp.json')
