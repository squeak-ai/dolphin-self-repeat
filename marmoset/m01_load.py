"""m01 — build calls.csv (all contexts) and calls2.csv (two-animal calls joined to Fig_2 spectral features).
Prints dataset summary and the version audit."""
import numpy as np, pandas as pd, os, glob
import scipy.io as sio
from m00_lib import *

df = load_calls(cache=False)
F = load_features()
res = {}

# ---- version audit (two_monkey: v1 vs v2 call totals)
diff = []
for d in sorted(glob.glob(RAW + '/two_monkey/*')):
    v1, v2 = os.path.join(d, 'call_data_v1.mat'), os.path.join(d, 'call_data_v2.mat')
    n = {}
    for tag, p in [('v1', v1), ('v2', v2)]:
        if os.path.exists(p):
            a = sio.loadmat(p)['call_data']
            n[tag] = sum(np.asarray(a[0, i]).reshape(-1, 2).shape[0] for i in range(a.shape[1]))
    diff.append(dict(key=os.path.basename(d), **n))
dd = pd.DataFrame(diff)
res['two_monkey_versions'] = dict(v1_total=int(dd.v1.sum()), v2_total=int(dd.v2.fillna(0).sum()), sessions=len(dd), v2_missing=int(dd.v2.isna().sum()))
print('two_monkey v1 total', dd.v1.sum(), '| v2 total', dd.v2.sum(), '| Fig_2 rows', len(F))

# ---- match Fig_2 sessions to .mat sessions by start-time sets, attach caller identity + features
two = df[df.ctx == 2]
ms = {s: g for s, g in two.groupby('session')}
rows = []
res['feature_match'] = {}
for s, g in F.groupby('Session'):
    a = np.sort(g['Start time'].values)
    best = None
    for t, b in ms.items():
        if len(b) != len(a): continue
        d = np.abs(a - np.sort(b.t0.values)).max()
        if best is None or d < best[1]: best = (t, d)
    if best is None or best[1] > 0.05:
        res['feature_match'][int(s)] = None; continue
    res['feature_match'][int(s)] = int(best[0])
    m = ms[best[0]].sort_values('t0').reset_index(drop=True)
    g = g.sort_values('Start time').reset_index(drop=True)
    j = pd.concat([m, g.drop(columns=['Session', 'Monkey 1', 'Monkey 2', 'Start time']).add_prefix('f_')], axis=1)
    rows.append(j)
c2 = pd.concat(rows).sort_values(['session', 't0']).reset_index(drop=True)
c2.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').lower() for c in c2.columns]
c2.to_csv(os.path.join(OUT, 'calls2.csv'), index=False)
print('calls2.csv', c2.shape, 'sessions', c2.session.nunique(), 'unmatched feature sessions', sum(v is None for v in res['feature_match'].values()))
# sanity: Fig_2 call length vs our (t1-t0)
print('call length agreement: max |diff| =', np.abs(c2.f_call_length - c2.dur).max())

# ---- session table
sess = df.groupby('session').agg(ctx=('ctx', 'first'), key=('key', 'first'), date=('date', 'first'), n=('idx', 'size'),
                                 callers=('caller', 'nunique'), T=('t1', 'max')).reset_index()
sess.to_csv(os.path.join(OUT, 'sessions.csv'), index=False)

# ---- summary
s = {}
for ctx, g in df.groupby('ctx'):
    ss = sess[sess.ctx == ctx]
    s[int(ctx)] = dict(sessions=int(len(ss)), calls=int(len(g)), animals=int(g.caller.nunique()),
                       dur_median=float(g.dur.median()), dur_iqr=[float(g.dur.quantile(.25)), float(g.dur.quantile(.75))],
                       dur_gt10s=int((g.dur > 10).sum()), pulses_median=float(g.pulses.median()),
                       session_T_median=float(ss['T'].median()), calls_per_min_median=float((ss.n / (ss['T'] / 60)).median()),
                       calls_per_session_median=float(ss.n.median()))
res['summary'] = s
res['animals_total'] = int(df.caller.nunique()); res['fps'] = FPS
for k, v in s.items(): print(k, v)
print('animals total', df.caller.nunique(), '| calls total', len(df))
dump('m01_load', res)
