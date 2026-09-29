"""d04c: triangle statistic for the lead figure — exact and transposed triples, both nulls, per-session counts.

Same construction as d03_triangles.py / d03_triangles2.py: three modulated whistles within 10 s (first-to-last onset),
pairwise non-overlapping, all three pairs 'exact' (DTW-abs <= p5) or all three 'transposed' (DTW-shape <= p5 and
|df_mean| > 1 kHz for every pair). Nulls: onset permutation within session and within 1-minute file, 200 draws each.
Output feeds article_fig1. Also records example exact triples (for article_fig5).
"""
import numpy as np, pandas as pd, json, time
from d01_lib import *

NDRAW = 200
df, F = load(); rng = np.random.default_rng(77)
Da = np.load(os.path.join(OUT, 'dtw_abs.npy')); Ds = np.load(os.path.join(OUT, 'dtw_shape.npy'))
mod = (df.cls != 'constant').values
sess = {s: d.index.values[mod[d.index.values]] for s, d in df.groupby('session')}
sess = {s: ix for s, ix in sess.items() if len(ix) >= 20}
allI = np.concatenate([ix[np.triu_indices(len(ix), 1)[0]] for ix in sess.values()])
allJ = np.concatenate([ix[np.triu_indices(len(ix), 1)[1]] for ix in sess.values()])
thr_a = np.percentile(Da[allI, allJ], 5); thr_s = np.percentile(Ds[allI, allJ], 5)
fm = df.f_mean.values / 1000
t0 = df.abs_t0.values; t1 = df.abs_t1.values
file_groups = [d.index.values for _, d in df[mod].groupby('file')]


def pair_ok(kind, a, b):
    if kind == 'exact': return Da[a, b] <= thr_a
    return Ds[a, b] <= thr_s and abs(fm[a] - fm[b]) > 1


def count(t0v, t1v, kind, collect=False):
    per = {}; tris = []
    for s, ix in sess.items():
        order = ix[np.argsort(t0v[ix])]; ts = t0v[order]; te = t1v[order]; n = 0
        for a in range(len(order)):
            hi = np.searchsorted(ts, ts[a] + 10, side='right')
            for b in range(a + 1, hi):
                if ts[b] < te[a] or not pair_ok(kind, order[a], order[b]): continue
                for c in range(b + 1, hi):
                    if ts[c] < te[b] or not pair_ok(kind, order[a], order[c]) or not pair_ok(kind, order[b], order[c]): continue
                    n += 1
                    if collect: tris.append((int(order[a]), int(order[b]), int(order[c])))
        per[int(s)] = n
    return (per, tris) if collect else per


res = dict(thr_abs=float(thr_a), thr_shape=float(thr_s), n_draws=NDRAW, n_sessions=len(sess))
for kind in ['exact', 'shifted']:
    tic = time.time()
    obs_per, tris = count(t0, t1, kind, collect=True)
    r = dict(obs=int(sum(obs_per.values())), per_session_obs=obs_per)
    if kind == 'exact': r['triples'] = tris
    for name in ['session', 'file']:
        draws = []
        for _ in range(NDRAW):
            t0p = t0.copy(); t1p = t1.copy()
            groups = file_groups if name == 'file' else list(sess.values())
            for ix in groups:
                perm = rng.permutation(len(ix)); t0p[ix] = t0[ix][perm]; t1p[ix] = t1[ix][perm]
            draws.append(count(t0p, t1p, kind))
        tot = np.array([sum(d.values()) for d in draws], float)
        per_null = {s: float(np.mean([d[s] for d in draws])) for s in obs_per}
        tested = [s for s in obs_per if per_null[s] > 0 or obs_per[s] > 0]
        r[name] = dict(null_mean=float(tot.mean()), null_sd=float(tot.std()), ratio=r['obs'] / tot.mean(), z=(r['obs'] - tot.mean()) / tot.std(),
                       per_session_null=per_null, sessions_gt=int(sum(obs_per[s] > per_null[s] for s in tested)), sessions_tested=len(tested))
        print(f'{kind:8s} {name:8s}: obs {r["obs"]} vs {tot.mean():.1f} ± {tot.std():.1f} -> {r["obs"]/tot.mean():.2f}x z {(r["obs"]-tot.mean())/tot.std():.1f}; sessions obs>null {r[name]["sessions_gt"]}/{len(tested)}')
    res[kind] = r
    print(f'  ({time.time()-tic:.0f}s)')
json.dump(res, open(os.path.join(OUT, 'd04_triangles.json'), 'w'), indent=1, default=float)
print('saved d04_triangles.json')
