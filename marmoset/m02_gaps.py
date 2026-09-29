"""m02 — turn-taking timing (dolphin findings §2 replicated on marmosets).

(1) Pooled gap distribution (gap = next onset − latest offset so far) vs two dolphin-style nulls:
    A: shuffle IOIs and durations independently within session (burst-preserving null, 07_gap_null.py)
    B: shuffle (duration, next IOI) pairs jointly (the Fable-verification null that retracted the dolphin excess)
    Caller sequence is kept, so gaps split into other-caller transitions vs same-caller transitions.
(2) Response latency with a SHIFT null (only possible with caller identity): for each other-caller transition
    A→B, latency = B onset − A offset. Null: circularly shift each caller's train by a random offset within the
    session (preserves every animal's own temporal structure, destroys only cross-animal coupling). 200 draws.
(3) Caller alternation: P(next caller ≠ current) vs shift null.
Sessions: all two- and three-animal sessions with ≥ 20 calls.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from m00_lib import *

rng = np.random.default_rng(7)
df = load_calls()
sess = pd.read_csv(os.path.join(OUT, 'sessions.csv'))
use = sess[(sess.ctx >= 2) & (sess.n >= 20)].session.values
print('sessions used', len(use), 'calls', df[df.session.isin(use)].shape[0])
EDGES = np.array([-np.inf, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 20, 60, np.inf])
LAB = ['<-0.5', '-0.5–0', '0–0.5', '0.5–1', '1–2', '2–3', '3–5', '5–10', '10–20', '20–60', '>60']
NIT = 200


def gaps_of(t0, t1, caller):
    """gap to latest offset so far; transition type other/same relative to the caller of the latest-offset call."""
    o = np.argsort(t0); t0, t1, caller = t0[o], t1[o], caller[o]
    g = np.full(len(t0), np.nan); typ = np.zeros(len(t0), dtype=int)  # 1 = other, 0 = same
    mx = t1[0]; mxc = caller[0]
    for i in range(1, len(t0)):
        g[i] = t0[i] - mx; typ[i] = int(caller[i] != mxc)
        if t1[i] > mx: mx, mxc = t1[i], caller[i]
    return g[1:], typ[1:]


def null_A(t0, t1, caller):
    o = np.argsort(t0); t0, t1, caller = t0[o], t1[o], caller[o]
    io = rng.permutation(np.diff(t0)); d = rng.permutation(t1 - t0)
    n0 = np.concatenate([[t0[0]], t0[0] + np.cumsum(io)])
    return n0, n0 + d, caller


def null_B(t0, t1, caller):
    o = np.argsort(t0); t0, t1, caller = t0[o], t1[o], caller[o]
    d = t1 - t0; io = np.diff(t0)
    p = rng.permutation(len(io))  # shuffle (dur_i, ioi_i) pairs jointly for i < n-1
    io2 = io[p]; d2 = np.concatenate([d[:-1][p], [d[-1]]])
    n0 = np.concatenate([[t0[0]], t0[0] + np.cumsum(io2)])
    return n0, n0 + d2, caller


def shift_null(t0, t1, caller, T):
    n0, n1 = t0.copy(), t1.copy()
    for c in np.unique(caller):
        m = caller == c; s = rng.uniform(30, T - 30)
        n0[m] = (t0[m] + s) % T; n1[m] = n0[m] + (t1[m] - t0[m])
    return n0, n1, caller


def hist(g, typ):
    h = {}
    for k, name in [(None, 'all'), (1, 'other'), (0, 'same')]:
        gg = g if k is None else g[typ == k]
        h[name] = np.histogram(gg, EDGES)[0]
    return h


res = {'bins': LAB, 'sessions': [int(s) for s in use]}
obs = {k: np.zeros(len(LAB)) for k in ['all', 'other', 'same']}
nulls = {nm: {k: np.zeros((NIT, len(LAB))) for k in obs} for nm in ['A', 'B', 'shift']}
per_sess = []
for s in use:
    d = df[df.session == s]; t0, t1, c = d.t0.values, d.t1.values, d.caller.values; T = float(sess[sess.session == s]['T'].iloc[0]) + 1
    g, typ = gaps_of(t0, t1, c); h = hist(g, typ)
    for k in obs: obs[k] += h[k]
    sn = {nm: {k: np.zeros((NIT, len(LAB))) for k in obs} for nm in nulls}
    for it in range(NIT):
        for nm, fn in [('A', null_A), ('B', null_B), ('shift', lambda a, b, cc: shift_null(a, b, cc, T))]:
            n0, n1, nc = fn(t0, t1, c); gn, tn = gaps_of(n0, n1, nc); hn = hist(gn, tn)
            for k in obs: nulls[nm][k][it] += hn[k]; sn[nm][k][it] = hn[k]
    # per-session: other-caller transitions in 0–5 s vs shift null; alternation rate
    b05 = slice(2, 7)  # 0–5 s bins
    per_sess.append(dict(session=int(s), ctx=int(d.ctx.iloc[0]), n=len(d), other_0_5_obs=int(h['other'][b05].sum()),
                         other_0_5_shift=float(sn['shift']['other'][:, b05].sum(1).mean()),
                         other_0_5_A=float(sn['A']['other'][:, b05].sum(1).mean()),
                         alt_obs=float(typ.mean()), alt_shift=float(np.mean([hist(*gaps_of(*shift_null(t0, t1, c, T)))['other'].sum() / (len(t0) - 1) for _ in range(50)])),
                         overlap_obs=int(h['other'][:2].sum()), overlap_shift=float(sn['shift']['other'][:, :2].sum(1).mean())))
ps = pd.DataFrame(per_sess); ps.to_csv(os.path.join(OUT, 'm02_per_session.csv'), index=False)


def table(k, nm):
    o = obs[k]; n = nulls[nm][k]; mu, sd = n.mean(0), n.std(0)
    return pd.DataFrame(dict(bin=LAB, obs=o.astype(int), null=mu.round(1), sd=sd.round(1), ratio=(o / np.where(mu > 0, mu, np.nan)).round(3), z=((o - mu) / np.where(sd > 0, sd, np.nan)).round(1)))


res['tables'] = {}
for k in ['all', 'other', 'same']:
    for nm in ['A', 'B', 'shift']:
        t = table(k, nm); res['tables'][f'{k}_{nm}'] = t.to_dict('list')
for k in ['other', 'same']:
    print(f'\n== {k}-caller transitions: obs | null A (IOI+dur shuffle) | null B (paired) | shift null')
    tA, tB, tS = table(k, 'A'), table(k, 'B'), table(k, 'shift')
    for i in range(len(LAB)):
        print(f'  {LAB[i]:>8} s  obs {int(obs[k][i]):6d} | A {tA.null[i]:8.1f} r {tA.ratio[i]:5.2f} z {tA.z[i]:6.1f} | B {tB.null[i]:8.1f} r {tB.ratio[i]:5.2f} z {tB.z[i]:6.1f} | S {tS.null[i]:8.1f} r {tS.ratio[i]:5.2f} z {tS.z[i]:6.1f}')

# per-session replication
r = ps.other_0_5_obs / ps.other_0_5_shift
w = wilcoxon(ps.other_0_5_obs - ps.other_0_5_shift).pvalue
res['per_session'] = dict(n=len(ps), ratio_gt1=int((r > 1).sum()), ratio_median=float(r.median()), wilcoxon_p=float(w),
                          alt_obs_mean=float(ps.alt_obs.mean()), alt_shift_mean=float(ps.alt_shift.mean()),
                          alt_gt_null=int((ps.alt_obs > ps.alt_shift).sum()), alt_wilcoxon_p=float(wilcoxon(ps.alt_obs - ps.alt_shift).pvalue),
                          overlap_obs=int(ps.overlap_obs.sum()), overlap_shift=float(ps.overlap_shift.sum()),
                          overlap_sessions_below=int((ps.overlap_obs < ps.overlap_shift).sum()))
print('\nper-session other-caller 0–5 s vs shift null: ratio>1 in', res['per_session']['ratio_gt1'], '/', len(ps), 'median', round(r.median(), 2), 'wilcoxon p', f'{w:.2g}')
print('alternation P(next≠current): obs', round(ps.alt_obs.mean(), 3), 'shift null', round(ps.alt_shift.mean(), 3), '| sessions above null', res['per_session']['alt_gt_null'], 'p', f"{res['per_session']['alt_wilcoxon_p']:.2g}")
print('other-caller overlaps (gap<0): obs', res['per_session']['overlap_obs'], 'shift null', round(res['per_session']['overlap_shift'], 1), '| sessions below null', res['per_session']['overlap_sessions_below'])

# ---- fine-resolution response latency (other-caller transitions), 0.25 s bins to 12 s, obs vs shift null — for the figure
FE = np.arange(-3, 12.01, 0.25)
fo = np.zeros(len(FE) - 1); fn = np.zeros((NIT, len(FE) - 1))
for s in use:
    d = df[df.session == s]; t0, t1, c = d.t0.values, d.t1.values, d.caller.values; T = float(sess[sess.session == s]['T'].iloc[0]) + 1
    g, typ = gaps_of(t0, t1, c); fo += np.histogram(g[typ == 1], FE)[0]
    for it in range(NIT):
        gn, tn = gaps_of(*shift_null(t0, t1, c, T)); fn[it] += np.histogram(gn[tn == 1], FE)[0]
res['fine'] = dict(edges=FE.tolist(), obs=fo.tolist(), null_mean=fn.mean(0).tolist(), null_sd=fn.std(0).tolist())
ratio = fo / fn.mean(0); pk = np.argmax(ratio[12:]) + 12
print(f'\nfine latency: peak ratio {ratio[pk]:.2f} at {FE[pk]:.2f}–{FE[pk+1]:.2f} s; ratio by 1-s: ', [round(float(ratio[12 + 4 * i:16 + 4 * i].mean()), 2) for i in range(10)])
# latency where ratio first exceeds 1 and where it returns to 1
above = np.where(ratio[12:] > 1)[0]
res['fine_summary'] = dict(peak_ratio=float(ratio[pk]), peak_bin=[float(FE[pk]), float(FE[pk + 1])],
                           first_above_1=float(FE[12 + above[0]]) if len(above) else None)
dump('m02_gaps', res)
