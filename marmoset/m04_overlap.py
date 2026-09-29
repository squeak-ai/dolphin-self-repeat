"""m04 — overlap / interruption (dolphin findings §4 + discovery §4 "no jamming avoidance").

(1) Overlapping other-caller pairs per session vs (a) shift null (each caller's train circularly shifted),
    (b) uniform placement of onsets (durations kept). Two- and three-animal sessions.
(2) Where do overlaps happen? Onset-to-onset offset of the overlapping pair: collisions (|Δonset| < 0.5 s,
    both start into the same silence) vs true interruptions (B starts while A is well underway). Observed vs shift null.
(3) Frequency avoidance during overlap (two-animal sessions with features): |Δ f_max| and |Δ f_end| of the
    overlapping pair vs null pairs = same A call paired with B calls from the same session within ±120 s
    (controls the slow drift of m03b). Also vs non-overlapping adjacent other-caller pairs.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from m00_lib import *

rng = np.random.default_rng(3)
df = load_calls(); sess = pd.read_csv(os.path.join(OUT, 'sessions.csv'))
c2 = pd.read_csv(os.path.join(OUT, 'calls2.csv'))
use = sess[(sess.ctx >= 2) & (sess.n >= 20)].session.values
NIT = 200
res = {}


def overlaps(t0, t1, c):
    """All other-caller pairs (i<j by onset) with t0[j] < t1[i]. Returns array of (i, j, onset_diff, overlap_len)."""
    o = np.argsort(t0); t0, t1, c = t0[o], t1[o], c[o]
    out = []
    for i in range(len(t0)):
        j = i + 1
        while j < len(t0) and t0[j] < t1[i]:
            if c[j] != c[i]: out.append((o[i], o[j], t0[j] - t0[i], min(t1[i], t1[j]) - t0[j]))
            j += 1
    return np.array(out).reshape(-1, 4)


def shift(t0, t1, c, T):
    n0 = t0.copy()
    for cc in np.unique(c):
        m = c == cc; n0[m] = (t0[m] + rng.uniform(30, T - 30)) % T
    return n0, n0 + (t1 - t0)


def uniform(t0, t1, c, T):
    n0 = rng.uniform(0, T, len(t0)); return n0, n0 + (t1 - t0)


rows = []; coll_obs = 0; coll_null = np.zeros(NIT); n_obs = 0; n_null_s = np.zeros(NIT); n_null_u = np.zeros(NIT)
onset_diffs_obs = []; onset_diffs_null = []
for s in use:
    d = df[df.session == s]; t0, t1, c = d.t0.values, d.t1.values, d.caller.values; T = float(sess[sess.session == s]['T'].iloc[0]) + 1
    ov = overlaps(t0, t1, c); n_obs += len(ov); onset_diffs_obs.append(ov[:, 2]); coll_obs += (ov[:, 2] < 0.5).sum()
    ns, nu, nc = np.zeros(NIT), np.zeros(NIT), np.zeros(NIT)
    for it in range(NIT):
        a, b = shift(t0, t1, c, T); o2 = overlaps(a, b, c); ns[it] = len(o2); nc[it] = (o2[:, 2] < 0.5).sum()
        if it < 20: onset_diffs_null.append(o2[:, 2])
        a, b = uniform(t0, t1, c, T); nu[it] = len(overlaps(a, b, c))
    n_null_s += ns; n_null_u += nu; coll_null += nc
    rows.append(dict(session=int(s), ctx=int(d.ctx.iloc[0]), n=len(d), obs=len(ov), shift_null=ns.mean(), shift_sd=ns.std(), uniform=nu.mean(),
                     p_shift=perm_p(len(ov), ns, 'less')))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'm04_per_session.csv'), index=False)
res['overlap'] = dict(obs=int(n_obs), shift=float(n_null_s.mean()), shift_sd=float(n_null_s.std()), uniform=float(n_null_u.mean()),
                      ratio_shift=float(n_obs / n_null_s.mean()), ratio_uniform=float(n_obs / n_null_u.mean()),
                      sessions=len(R), sessions_below_shift=int((R.obs < R.shift_null).sum()), sessions_p05=int((R.p_shift < 0.05).sum()),
                      wilcoxon_p=float(wilcoxon(R.obs - R.shift_null).pvalue), sessions_with_any=int((R.obs > 0).sum()),
                      ncalls=int(df[df.session.isin(use)].shape[0]))
for ctx in [2, 3]:
    r = R[R.ctx == ctx]; res['overlap'][f'ctx{ctx}'] = dict(obs=int(r.obs.sum()), shift=float(r.shift_null.sum()), ratio=float(r.obs.sum() / r.shift_null.sum()), sessions=len(r), below=int((r.obs < r.shift_null).sum()),
                                                             ncalls=int(r.n.sum()), pct_calls=float(100 * r.obs.sum() / r.n.sum()))
o = res['overlap']
print(f"(1) overlapping other-caller pairs: obs {o['obs']} ({100*o['obs']/o['ncalls']:.2f}% of {o['ncalls']} calls) | shift null {o['shift']:.0f} ± {o['shift_sd']:.0f} → {o['ratio_shift']:.2f}× | uniform null {o['uniform']:.0f} → {o['ratio_uniform']:.2f}×")
print(f"    sessions below shift null {o['sessions_below_shift']}/{o['sessions']} (p<0.05 in {o['sessions_p05']}), Wilcoxon p={o['wilcoxon_p']:.1e}; sessions with ≥1 overlap {o['sessions_with_any']}")
print(f"    dyads: {o['ctx2']['obs']} obs vs {o['ctx2']['shift']:.0f} ({o['ctx2']['ratio']:.2f}×, {o['ctx2']['pct_calls']:.2f}% of calls) | trios: {o['ctx3']['obs']} vs {o['ctx3']['shift']:.0f} ({o['ctx3']['ratio']:.2f}×, {o['ctx3']['pct_calls']:.2f}%)")

# (2) collisions vs interruptions
od = np.concatenate(onset_diffs_obs); nd = np.concatenate(onset_diffs_null)
E = [0, 0.25, 0.5, 1, 2, 3, 5, 20]
ho = np.histogram(od, E)[0]; hn = np.histogram(nd, E)[0] / 20
res['onset_diff'] = dict(edges=E, obs=ho.tolist(), null=hn.tolist(), ratio=(ho / hn).tolist(),
                         collision_frac_obs=float((od < 0.5).mean()), collision_frac_null=float((nd < 0.5).mean()),
                         collision_ratio=float(coll_obs / coll_null.mean()), late_ratio=float((od >= 0.5).sum() / ((nd >= 0.5).sum() / 20)))
print(f"(2) onset-to-onset offset of overlapping pairs: obs hist {ho.tolist()} | null {np.round(hn).astype(int).tolist()} | ratio {np.round(ho/hn, 2).tolist()}")
print(f"    share that are collisions (<0.5 s): obs {res['onset_diff']['collision_frac_obs']:.2f} vs null {res['onset_diff']['collision_frac_null']:.2f}; collisions {res['onset_diff']['collision_ratio']:.2f}× null, later interruptions {res['onset_diff']['late_ratio']:.2f}× null")

# (3) frequency avoidance during overlap, two-animal sessions with features
c2 = c2.dropna(subset=['f_maximum_frequency', 'f_end_frequency']).copy()
dmax_o, dmax_n, dend_o, dend_n, dmax_adj = [], [], [], [], []
for s, g in c2.groupby('session'):
    g = g.sort_values('t0').reset_index(drop=True); t0, t1, c = g.t0.values, g.t1.values, g.caller.values
    ov = overlaps(t0, t1, c)
    fm, fe = g.f_maximum_frequency.values, g.f_end_frequency.values
    for i, j, _, _ in ov:
        i, j = int(i), int(j); dmax_o.append(abs(fm[i] - fm[j])); dend_o.append(abs(fe[i] - fe[j]))
        cand = np.where((c == c[j]) & (np.abs(t0 - t0[i]) <= 120) & (np.arange(len(g)) != j))[0]
        if len(cand):
            k = rng.choice(cand, size=min(len(cand), 20)); dmax_n.append(np.abs(fm[i] - fm[k]).mean()); dend_n.append(np.abs(fe[i] - fe[k]).mean())
        else:
            dmax_n.append(np.nan); dend_n.append(np.nan)
    # non-overlapping adjacent other-caller pairs with gap 0–5 s
    adj = np.where((c[1:] != c[:-1]) & (t0[1:] - t1[:-1] >= 0) & (t0[1:] - t1[:-1] < 5))[0]
    dmax_adj += list(np.abs(fm[adj] - fm[adj + 1]))
dmax_o, dmax_n, dend_o, dend_n = map(np.array, (dmax_o, dmax_n, dend_o, dend_n)); m = ~np.isnan(dmax_n)
res['freq_avoid'] = dict(n=int(m.sum()), dmax_obs_median=float(np.median(dmax_o[m])), dmax_null_median=float(np.median(dmax_n[m])),
                         dmax_wilcoxon_p=float(wilcoxon(dmax_o[m] - dmax_n[m]).pvalue), dend_obs_median=float(np.median(dend_o[m])), dend_null_median=float(np.median(dend_n[m])),
                         dend_wilcoxon_p=float(wilcoxon(dend_o[m] - dend_n[m]).pvalue), dmax_adjacent_median=float(np.median(dmax_adj)), n_adjacent=len(dmax_adj),
                         mwu_overlap_vs_adjacent_p=float(mannwhitneyu(dmax_o, dmax_adj).pvalue))
f = res['freq_avoid']
print(f"(3) frequency separation during overlap (n={f['n']} dyad overlaps): |Δf_max| median obs {f['dmax_obs_median']:.0f} vs local null {f['dmax_null_median']:.0f} bins (Wilcoxon p={f['dmax_wilcoxon_p']:.2g}); |Δf_end| {f['dend_obs_median']:.0f} vs {f['dend_null_median']:.0f} (p={f['dend_wilcoxon_p']:.2g})")
print(f"    vs non-overlapping adjacent pairs (gap 0–5 s, n={f['n_adjacent']}): median |Δf_max| {f['dmax_adjacent_median']:.0f}, MWU p={f['mwu_overlap_vs_adjacent_p']:.2g}")
dump('m04_overlap', res)
