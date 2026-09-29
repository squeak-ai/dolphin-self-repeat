"""m05 — calling as a point process (dolphin d01_pointprocess.py replicated on marmosets).

(1) Fano factor of onset counts across window sizes vs renewal (IOI-shuffle) null — pooled per session and
    per caller; absolute windows (0.25–60 s, dolphin grid) and rescaled windows (multiples of the median IOI,
    because marmosets call ~30× more slowly than the dolphin chorus).
(2) Burstiness B = (σ−μ)/(σ+μ) and memory M = corr(IOI_i, IOI_{i+1}) (Goh & Barabási 2008), per session and per caller,
    vs renewal null for M (B is invariant to IOI shuffling by construction).
(3) Hawkes (exponential kernel) MLE per session with ≥100 calls: branching ratio α/β, memory 1/β, vs renewal-null
    branching; time-rescaling KS for Hawkes / homogeneous Poisson / piecewise-constant (60-s) Poisson.
    Vectorised recursion via logaddexp.accumulate (no Python loop over events).
(4) Solo vs dyad vs trio on every statistic.
Dolphin reference (d01_pointprocess.json, sessions ≥100 whistles): Fano 1.07/1.43/4.06/10.9 at 0.25/1/10/60 s,
renewal 1.06/1.29/2.23/2.63; branching median 0.67, renewal-null branching 0.11–0.71 → Cox, not Hawkes; B = 0.42.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest, wilcoxon, mannwhitneyu
from m00_lib import *

rng = np.random.default_rng(11)
df = load_calls()
sess = pd.read_csv(os.path.join(OUT, 'sessions.csv')).set_index('session')
MIN_POOL, MIN_CALLER, MIN_HAWKES = 50, 30, 100
NNULL = 50
WIN = [0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180]
RESC = [0.25, 0.5, 1, 2, 5, 10]          # windows in units of median IOI
res = {}


def train(d):
    t = np.sort(d.t0.values); t = t - t[0]
    return t


def renewal(t):
    io = rng.permutation(np.diff(t)); return np.concatenate([[t[0]], t[0] + np.cumsum(io)])


def fano_w(t, T, w):
    nb = int(np.floor(T / w))
    if nb < 5: return np.nan
    c = np.histogram(t, bins=nb, range=(0, nb * w))[0]
    return c.var() / c.mean() if c.mean() > 0 else np.nan


def memory(iois):
    if len(iois) < 4: return np.nan
    return float(np.corrcoef(iois[:-1], iois[1:])[0, 1])


# ---------- unit sets ----------
units = []  # (label, ctx, session, caller|None, t, T)
for s, d in df.groupby('session'):
    T = float(sess.loc[s, 'T']); ctx = int(d.ctx.iloc[0])
    if len(d) >= MIN_POOL:
        units.append(dict(kind='pooled', ctx=ctx, session=int(s), caller=None, t=train(d), T=T))
    if ctx >= 2:
        for c, dc in d.groupby('caller'):
            if len(dc) >= MIN_CALLER:
                units.append(dict(kind='caller', ctx=ctx, session=int(s), caller=c, t=train(dc), T=T))
print('units: pooled', sum(u['kind'] == 'pooled' for u in units), 'per-caller', sum(u['kind'] == 'caller' for u in units))

# ---------- (1) Fano + (2) B, M ----------
rows = []
for u in units:
    t, T = u['t'], u['T']; io = np.diff(t); med = np.median(io)
    r = dict(kind=u['kind'], ctx=u['ctx'], session=u['session'], caller=u['caller'], n=len(t), med_ioi=med, rate=len(t) / T,
             B=burstiness(io), M=memory(io))
    nulls = [renewal(t) for _ in range(NNULL)]
    r['M_null'] = float(np.mean([memory(np.diff(x)) for x in nulls]))
    for w in WIN:
        r[f'F_{w}'] = fano_w(t, T, w); r[f'Fn_{w}'] = float(np.nanmean([fano_w(x, T, w) for x in nulls]))
    for k in RESC:
        w = k * med
        r[f'R_{k}'] = fano_w(t, T, w); r[f'Rn_{k}'] = float(np.nanmean([fano_w(x, T, w) for x in nulls]))
    rows.append(r)
pp = pd.DataFrame(rows); pp.to_csv(os.path.join(OUT, 'm05_units.csv'), index=False)


def summarise(sub, label):
    out = dict(n_units=int(len(sub)), n_calls=int(sub.n.sum()), rate_median=float(sub.rate.median()), med_ioi_median=float(sub.med_ioi.median()),
               B_mean=float(sub.B.mean()), B_median=float(sub.B.median()), M_mean=float(sub.M.mean()), M_null_mean=float(sub.M_null.mean()),
               M_gt_null=int((sub.M > sub.M_null).sum()), M_wilcoxon_p=float(wilcoxon(sub.M - sub.M_null).pvalue) if len(sub) > 5 else None)
    out['fano'] = {str(w): dict(real=float(sub[f'F_{w}'].mean()), renewal=float(sub[f'Fn_{w}'].mean()),
                                real_median=float(sub[f'F_{w}'].median()), ratio_median=float((sub[f'F_{w}'] / sub[f'Fn_{w}']).median()),
                                gt_null=int((sub[f'F_{w}'] > sub[f'Fn_{w}']).sum()), n=int(sub[f'F_{w}'].notna().sum())) for w in WIN}
    out['fano_rescaled'] = {str(k): dict(real=float(sub[f'R_{k}'].mean()), renewal=float(sub[f'Rn_{k}'].mean()),
                                         ratio_median=float((sub[f'R_{k}'] / sub[f'Rn_{k}']).median())) for k in RESC}
    print(f'\n== {label}: {out["n_units"]} units, {out["n_calls"]} calls, rate {out["rate_median"]*60:.1f}/min, median IOI {out["med_ioi_median"]:.1f} s, '
          f'B mean {out["B_mean"]:.2f} (median {out["B_median"]:.2f}), M {out["M_mean"]:+.3f} vs renewal {out["M_null_mean"]:+.3f} '
          f'({out["M_gt_null"]}/{out["n_units"]} above, p={out["M_wilcoxon_p"]})')
    print('   Fano window: real / renewal (units > null)')
    for w in WIN:
        f = out['fano'][str(w)]; print(f'     {w:6.2f} s  {f["real"]:6.2f} / {f["renewal"]:6.2f}   ({f["gt_null"]}/{f["n"]})')
    print('   Fano rescaled (× median IOI):', {k: (round(v['real'], 2), round(v['renewal'], 2)) for k, v in out['fano_rescaled'].items()})
    return out


res['pooled'] = {}
res['pooled']['all'] = summarise(pp[pp.kind == 'pooled'], 'POOLED all contexts')
for c, nm in [(1, 'solo'), (2, 'dyad'), (3, 'trio')]:
    res['pooled'][nm] = summarise(pp[(pp.kind == 'pooled') & (pp.ctx == c)], f'POOLED {nm}')
res['caller'] = {}
res['caller']['all'] = summarise(pp[pp.kind == 'caller'], 'PER-CALLER (dyads+trios)')
for c, nm in [(2, 'dyad'), (3, 'trio')]:
    res['caller'][nm] = summarise(pp[(pp.kind == 'caller') & (pp.ctx == c)], f'PER-CALLER {nm}')

# context contrasts on B and Fano60 (pooled)
pool = pp[pp.kind == 'pooled']
ctr = {}
for a, b in [(1, 2), (2, 3), (1, 3)]:
    x, y = pool[pool.ctx == a], pool[pool.ctx == b]
    ctr[f'{a}v{b}'] = dict(B_p=float(mannwhitneyu(x.B, y.B).pvalue), F60_p=float(mannwhitneyu(x.F_60.dropna(), y.F_60.dropna()).pvalue),
                          F10_p=float(mannwhitneyu(x.F_10.dropna(), y.F_10.dropna()).pvalue), rate_p=float(mannwhitneyu(x.rate, y.rate).pvalue))
res['context_contrasts'] = ctr
print('\ncontext contrasts (MWU p): ', {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ctr.items()})
# per-caller B in solo (the animal alone) vs same animal's per-caller B in dyads
solo_B = pool[pool.ctx == 1].set_index('session').B
res['B_solo_vs_caller_dyad'] = dict(solo_mean=float(pool[pool.ctx == 1].B.mean()), caller_dyad_mean=float(pp[(pp.kind == 'caller') & (pp.ctx == 2)].B.mean()),
                                    p=float(mannwhitneyu(pool[pool.ctx == 1].B, pp[(pp.kind == 'caller') & (pp.ctx == 2)].B).pvalue))
print('B: solo animals', round(res['B_solo_vs_caller_dyad']['solo_mean'], 3), 'vs individual callers inside dyads', round(res['B_solo_vs_caller_dyad']['caller_dyad_mean'], 3), 'p', f"{res['B_solo_vs_caller_dyad']['p']:.3g}")


# ---------- (3) Hawkes ----------
def hawkes_R(t, b):
    """R_i = sum_{j<i} exp(-b (t_i - t_j)), vectorised: exp(logcumsumexp(b t_j)_{i-1} - b t_i)."""
    lc = np.logaddexp.accumulate(b * t)
    R = np.zeros(len(t)); R[1:] = np.exp(lc[:-1] - b * t[1:])
    return R


BOUNDS = [(np.log(1e-5), np.log(2)), (np.log(1e-7), np.log(5)), (np.log(1e-3), np.log(20))]   # log mu, log alpha, log beta


def hawkes_nll(p, t, T):
    mu, a, b = np.exp(np.clip(p, [lo for lo, _ in BOUNDS], [hi for _, hi in BOUNDS]))
    R = hawkes_R(t, b); lam = mu + a * R
    v = -(np.sum(np.log(lam)) - mu * T - (a / b) * np.sum(1 - np.exp(-b * (T - t))))
    return v if np.isfinite(v) else 1e12


def hawkes_fit(t, T):
    best = None
    for b0 in [0.01, 0.1, 1]:
        for a0 in [0.001, 0.05]:
            r = minimize(hawkes_nll, np.log([len(t) / T / 2, a0, b0]), args=(t, T), method='L-BFGS-B', bounds=BOUNDS)
            if best is None or r.fun < best.fun: best = r
    mu, a, b = np.exp(best.x); return mu, a, b, -best.fun


def ks_rescaled(L):
    return kstest(np.diff(L), 'expon').statistic


hk = []
hs = [u for u in units if u['kind'] == 'pooled' and len(u['t']) >= MIN_HAWKES]
print(f'\n(3) Hawkes fits, {len(hs)} pooled sessions ≥{MIN_HAWKES} calls: ctx session n mu alpha beta branching memory KS[hawkes/poisson/pw60]  renewal-null branching')
for u in hs:
    t, T = u['t'], u['T']
    mu, a, b, ll = hawkes_fit(t, T)
    R = hawkes_R(t, b); L = mu * t + (a / b) * (np.arange(len(t)) - R); ks_h = ks_rescaled(L)
    ks_p = ks_rescaled(t * len(t) / T)
    edges = np.arange(0, T + 60, 60); c = np.histogram(t, edges)[0]; rate = c / 60
    Lpw = np.array([np.sum(rate[:int(x // 60)]) * 60 + rate[min(int(x // 60), len(rate) - 1)] * (x % 60) for x in t]); ks_pw = ks_rescaled(Lpw)
    nb = [(lambda f: f[1] / f[2])(hawkes_fit(renewal(t), T)) for _ in range(10)]
    hk.append(dict(ctx=u['ctx'], session=u['session'], n=len(t), mu=mu, alpha=a, beta=b, branching=a / b, memory=1 / b,
                   ks_hawkes=ks_h, ks_poisson=ks_p, ks_pw60=ks_pw, br_null_mean=float(np.mean(nb)), br_null_sd=float(np.std(nb))))
    print(f'    {u["ctx"]} {u["session"]:4d} {len(t):4d} {mu:6.4f} {a:6.4f} {b:7.4f}  {a/b:5.2f} {1/b:7.1f}  {ks_h:.3f}/{ks_p:.3f}/{ks_pw:.3f}   {np.mean(nb):.2f}±{np.std(nb):.2f}')
hk = pd.DataFrame(hk); hk.to_csv(os.path.join(OUT, 'm05_hawkes.csv'), index=False)
res['hawkes'] = {}
for c, nm in [(0, 'all'), (1, 'solo'), (2, 'dyad'), (3, 'trio')]:
    h = hk if c == 0 else hk[hk.ctx == c]
    if len(h) == 0: continue
    res['hawkes'][nm] = dict(n=int(len(h)), branching_median=float(h.branching.median()), branching_iqr=[float(h.branching.quantile(.25)), float(h.branching.quantile(.75))],
                             memory_median=float(h.memory.median()), memory_iqr=[float(h.memory.quantile(.25)), float(h.memory.quantile(.75))],
                             br_null_median=float(h.br_null_mean.median()), br_null_range=[float(h.br_null_mean.min()), float(h.br_null_mean.max())],
                             br_gt_null=int((h.branching > h.br_null_mean + 2 * h.br_null_sd).sum()),
                             ks=dict(hawkes=float(h.ks_hawkes.median()), poisson=float(h.ks_poisson.median()), pw60=float(h.ks_pw60.median())),
                             hawkes_beats_pw60=int((h.ks_hawkes < h.ks_pw60).sum()))
    r = res['hawkes'][nm]
    print(f'   {nm}: n={r["n"]} branching median {r["branching_median"]:.2f} IQR {r["branching_iqr"][0]:.2f}–{r["branching_iqr"][1]:.2f}, memory median {r["memory_median"]:.1f} s, '
          f'renewal-null branching median {r["br_null_median"]:.2f} (range {r["br_null_range"][0]:.2f}–{r["br_null_range"][1]:.2f}), >null+2sd in {r["br_gt_null"]}; '
          f'KS median hawkes {r["ks"]["hawkes"]:.3f} / poisson {r["ks"]["poisson"]:.3f} / pw60 {r["ks"]["pw60"]:.3f}; hawkes<pw60 in {r["hawkes_beats_pw60"]}')

res['dolphin_reference'] = dict(fano={'0.25': [1.07, 1.06], '1': [1.43, 1.29], '10': [4.06, 2.23], '60': [10.87, 2.63]}, B=0.42, branching_median=0.67,
                                memory_median_s=2.5, note='d01_pointprocess.json, sessions >= 100 whistles; Cox not Hawkes')
dump('m05_pointprocess', res)

# ---------- figure ----------
f, plt = fig(13, 4.6)
gs = f.add_gridspec(1, 3, wspace=0.32, top=0.82)
ax = f.add_subplot(gs[0, 0]); style(ax, 'a. Fano factor, absolute windows', 'window (s), log', 'var / mean of counts, log')
for nm, col, ls in [('solo', GREY, '-'), ('dyad', MARMOSET, '-'), ('trio', GREEN, '-')]:
    fr = [res['pooled'][nm]['fano'][str(w)]['real'] for w in WIN]; fn = [res['pooled'][nm]['fano'][str(w)]['renewal'] for w in WIN]
    ax.plot(WIN, fr, 'o-', color=col, ms=3.5, lw=1.4, label=f'marmoset {nm}')
    ax.plot(WIN, fn, '--', color=col, lw=0.9, alpha=0.6)
dw = [0.25, 0.5, 1, 2, 5, 10, 20, 30, 60]
dr = [1.07, 1.19, 1.43, 1.85, 2.89, 4.06, 5.49, 7.14, 10.87]; dn = [1.06, 1.14, 1.29, 1.53, 1.91, 2.23, 2.42, 2.45, 2.63]
ax.plot(dw, dr, 'o-', color=DOLPHIN, ms=3.5, lw=1.4, label='dolphin chorus'); ax.plot(dw, dn, '--', color=DOLPHIN, lw=0.9, alpha=0.6)
ax.axhline(1, color=GRID, lw=1); ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(frameon=False, fontsize=7.5)
ax.text(0.98, 0.03, 'dashed = renewal (IOI-shuffle) null', transform=ax.transAxes, fontsize=7, color=INK2, va='bottom', ha='right')
ax = f.add_subplot(gs[0, 1]); style(ax, 'b. Fano factor, windows in median IOIs', 'window / median IOI, log', 'var / mean of counts')
for nm, col in [('solo', GREY), ('dyad', MARMOSET), ('trio', GREEN)]:
    fr = [res['pooled'][nm]['fano_rescaled'][str(k)]['real'] for k in RESC]; fn = [res['pooled'][nm]['fano_rescaled'][str(k)]['renewal'] for k in RESC]
    ax.plot(RESC, fr, 'o-', color=col, ms=3.5, lw=1.4, label=f'marmoset {nm}'); ax.plot(RESC, fn, '--', color=col, lw=0.9, alpha=0.6)
# dolphin: median IOI 0.62 s -> windows 0.25/0.5/1/2/5/10 s ≈ 0.4/0.8/1.6/3.2/8/16 median IOIs
dm = 0.6217; ax.plot(np.array(dw) / dm, dr, 'o-', color=DOLPHIN, ms=3.5, lw=1.4, label='dolphin chorus'); ax.plot(np.array(dw) / dm, dn, '--', color=DOLPHIN, lw=0.9, alpha=0.6)
ax.axhline(1, color=GRID, lw=1); ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(frameon=False, fontsize=7.5)
ax = f.add_subplot(gs[0, 2]); style(ax, 'c. Burstiness B and memory M per session', 'burstiness B = (σ−μ)/(σ+μ)', 'memory M = corr(IOIᵢ, IOIᵢ₊₁)')
for nm, c, col in [('solo', 1, GREY), ('dyad', 2, MARMOSET), ('trio', 3, GREEN)]:
    sub = pool[pool.ctx == c]; ax.scatter(sub.B, sub.M, s=12, color=col, alpha=0.7, edgecolor='none', label=f'marmoset {nm} (n={len(sub)})')
ax.axvline(0.42, color=DOLPHIN, lw=1.2, ls=':', label='dolphin B = 0.42')
ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1); ax.set_xlim(-0.7, 0.6); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
f.suptitle('Marmoset phee calls as a point process — Grijseels et al. 2024, pooled sessions with ≥ 50 calls (dolphin: DOLPHINFREE sessions ≥ 100 whistles)', x=0.01, y=0.98, ha='left', fontsize=11, color=INK)
save(f, 'pointprocess')
