"""d04 shared pair table + permutation helpers (pre-article analyses).

Pair unit = every unordered pair of modulated whistles within a session that has >= 20 modulated whistles
(d02 / d03 convention). Vectorised so that a permutation null with 50 draws runs in seconds.
"""
import numpy as np
from d01_lib import *


class Pairs:
    def __init__(self, df, Da, Ds, min_session=20):
        mod = (df.cls != 'constant').values
        self.df = df; self.mod = mod; self.n_mod = int(mod.sum())
        self.t0 = df.abs_t0.values; self.t1 = df.abs_t1.values; self.fm = df.f_mean.values / 1000
        PI, PJ, PS = [], [], []; self.sess_ix = {}
        for s, d in df.groupby('session'):
            ix = d.index.values[mod[d.index.values]]
            if len(ix) < min_session: continue
            self.sess_ix[s] = ix
            I, J = np.triu_indices(len(ix), 1)
            PI.append(ix[I]); PJ.append(ix[J]); PS.append(np.full(len(I), s))
        self.I = np.concatenate(PI); self.J = np.concatenate(PJ); self.S = np.concatenate(PS)
        self.da = Da[self.I, self.J]; self.ds = Ds[self.I, self.J]; self.dfm = np.abs(self.fm[self.I] - self.fm[self.J])
        self.sess_pos = {s: np.where(self.S == s)[0] for s in sorted(self.sess_ix)}
        self.file_groups = [d.index.values for _, d in df[mod].groupby('file')]
        self.sess_groups = list(self.sess_ix.values())

    def thr(self, pct, which='abs'):
        return float(np.percentile(self.da if which == 'abs' else self.ds, pct))

    def geometry(self, t0v, t1v):
        a0 = t0v[self.I]; b0 = t0v[self.J]; first_i = a0 <= b0
        lag = np.abs(b0 - a0)
        first_end = np.where(first_i, t1v[self.I], t1v[self.J]); second_on = np.where(first_i, b0, a0)
        gap = second_on - first_end
        return lag, gap, gap < 0

    def permuted(self, rng, groups):
        t0p = self.t0.copy(); t1p = self.t1.copy()
        for ix in groups:
            perm = rng.permutation(len(ix)); t0p[ix] = self.t0[ix][perm]; t1p[ix] = self.t1[ix][perm]
        return t0p, t1p

    def participation(self, sel):
        return int(len(np.unique(np.concatenate([self.I[sel], self.J[sel]]))))
