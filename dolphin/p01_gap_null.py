"""p01 — gap histogram vs burst-preserving null A (inter-onset intervals and durations shuffled independently within
session, 200 draws). Tests turn-taking latency; output is the dolphin 'null A' curve used by m06_crossspecies.py.

Reads whistles_obs.csv (run p00_prepare.py first); writes gap_vs_null.csv (bin, obs, null, sd, z, ratio).
Gap = onset of each whistle minus the running maximum end time of all earlier whistles in the session
(negative = overlap).
"""
import os
import numpy as np, pandas as pd

OUT = os.environ.get('DOLPHIN_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dolphin'))
df = pd.read_csv(os.path.join(OUT, 'whistles_obs.csv'))
rng = np.random.default_rng(0)
bins = [-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2, 100]


def gaps_of(t0, t1):
    o = np.argsort(t0); t0, t1 = t0[o], t1[o]
    rm = np.maximum.accumulate(t1)
    return t0[1:] - rm[:-1]


obs, null = [], []
for s, g in df.groupby('session'):
    t0 = g.abs_t0.values; t1 = g.abs_t1.values; d = t1 - t0
    obs += list(gaps_of(t0, t1))
    if len(g) < 2:
        continue
    o = np.argsort(t0); ioi = np.diff(t0[o]); ds = d[o]
    for k in range(200):
        s0 = t0.min() + np.concatenate([[0], np.cumsum(rng.permutation(ioi))]); d2 = rng.permutation(ds)
        h = np.histogram(gaps_of(s0, s0 + d2), bins)[0]
        if k >= len(null): null.append(h)
        else: null[k] = null[k] + h
obs = np.histogram(np.array(obs), bins)[0]; null = np.array(null)
t = pd.DataFrame(dict(bin=[f'{a}..{b}' for a, b in zip(bins[:-1], bins[1:])], obs=obs, null=null.mean(0).round(1), sd=null.std(0).round(1)))
t['z'] = ((t.obs - t.null) / t.sd).round(2); t['ratio'] = (t.obs / t.null).round(3)
print(t.to_string())
t.to_csv(os.path.join(OUT, 'gap_vs_null.csv'), index=False)
