"""d03b: triangles — triples of modulated whistles, all three pairwise exact (DTW-abs <= p5), pairwise non-overlapping,
first-to-last onset lag <= 10 s. Split by the median of the two consecutive gaps (< 1 s vs >= 1 s), and a 'Janik recipe'
count: all consecutive gaps < 1 s. Null: onset permutation within session, 50 draws. Also the same for shape-matched but
pitch-shifted triples (DTW-shape <= p5 and |Δf_mean| > 1 kHz for every pair) as a control that should show no excess.
"""
import numpy as np, pandas as pd, json, itertools
from d01_lib import *
df, F = load(); rng = np.random.default_rng(5)
Da = np.load(os.path.join(OUT, 'dtw_abs.npy')); Ds = np.load(os.path.join(OUT, 'dtw_shape.npy'))
mod = (df.cls != 'constant').values
sess = {s: d.index.values[mod[d.index.values]] for s, d in df.groupby('session')}
sess = {s: ix for s, ix in sess.items() if len(ix) >= 20}
allI = []; allJ = []
for ix in sess.values():
    I, J = np.triu_indices(len(ix), 1); allI.append(ix[I]); allJ.append(ix[J])
allI = np.concatenate(allI); allJ = np.concatenate(allJ)
thr_a = np.percentile(Da[allI, allJ], 5); thr_s = np.percentile(Ds[allI, allJ], 5)
fm = df.f_mean.values / 1000
t0 = df.abs_t0.values; t1 = df.abs_t1.values

def count(t0v, t1v, kind):
    out = dict(n3=0, n3_med_lt1=0, n3_all_lt1=0, n3_med_1_10=0)
    for ix in sess.values():
        order = ix[np.argsort(t0v[ix])]
        ts = t0v[order]; te = t1v[order]
        for a in range(len(order)):
            # candidates within 10 s after a
            hi = np.searchsorted(ts, ts[a] + 10, side='right')
            for b in range(a + 1, hi):
                if ts[b] < te[a]: continue
                ia, ib = order[a], order[b]
                if kind == 'exact':
                    if Da[ia, ib] > thr_a: continue
                else:
                    if Ds[ia, ib] > thr_s or abs(fm[ia] - fm[ib]) <= 1: continue
                for c in range(b + 1, hi):
                    if ts[c] < te[b]: continue
                    ic = order[c]
                    if kind == 'exact':
                        if Da[ia, ic] > thr_a or Da[ib, ic] > thr_a: continue
                    else:
                        if Ds[ia, ic] > thr_s or Ds[ib, ic] > thr_s or abs(fm[ia]-fm[ic]) <= 1 or abs(fm[ib]-fm[ic]) <= 1: continue
                    g1 = ts[b] - te[a]; g2 = ts[c] - te[b]; med = np.median([g1, g2])
                    out['n3'] += 1
                    if med < 1: out['n3_med_lt1'] += 1
                    else: out['n3_med_1_10'] += 1
                    if g1 < 1 and g2 < 1: out['n3_all_lt1'] += 1
    return out

res = {}
for kind in ['exact', 'shifted']:
    obs = count(t0, t1, kind); null = []
    for _ in range(50):
        t0p = t0.copy(); t1p = t1.copy()
        for ix in sess.values():
            perm = rng.permutation(len(ix)); t0p[ix] = t0[ix][perm]; t1p[ix] = t1[ix][perm]
        null.append(count(t0p, t1p, kind))
    nd = pd.DataFrame(null)
    print(f'== {kind} triangles')
    for k in obs:
        m, sd = nd[k].mean(), nd[k].std()
        print(f'  {k:12s} obs {obs[k]:6d}  null {m:7.1f} ± {sd:5.1f}  ratio {obs[k]/m if m else np.nan:.2f}  z {(obs[k]-m)/sd if sd else np.nan:5.1f}')
    res[kind] = dict(obs=obs, null_mean=nd.mean().to_dict(), null_sd=nd.std().to_dict())
json.dump(res, open(os.path.join(OUT, 'd03_triangles.json'), 'w'), indent=1)
