"""m06 — cross-species figures: DOLPHINFREE common dolphins (single hydrophone, no caller identity) vs
Grijseels 2024 marmosets (caller identity known).

(a) marmoset_xs_gaps.png      — gap-to-previous-call ratio vs null. Left: absolute gap. Right: gap in units of the
                                 median call duration. Dolphin: null A (IOI+dur shuffle) and null B (paired shuffle);
                                 marmoset: pooled null A / null B (dolphin-style, no identity) and other-caller vs shift null.
(b) marmoset_xs_selfrepeat.png — exact self-repeat excess vs gap, both against within-minute-class nulls
                                 (dolphin d04 file null; marmoset m03c 60-s block null). Absolute and rescaled.
(c) marmoset_xs_table.png      — comparison table rendered as a figure.
Numbers are read from the JSON outputs of d04_renull, m02_gaps, m03c_exact_local, m04_overlap, m05_pointprocess and
from the dolphin analysis outputs (d04_renull.json, gap_vs_null.csv).
"""
import sys, textwrap
import numpy as np, pandas as pd
from m00_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from figstyle import *          # shared palette/type scale; shadows m00_lib.style


def fig(w=FULL_W, h=5):
    return plt.figure(figsize=(w, h), facecolor=SURF), plt


def save(f, name):
    save_both(f, FIG, f'marmoset_{name}')

D = os.environ.get('DOLPHIN_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'dolphin'))
m02 = json.load(open(os.path.join(OUT, 'm02_gaps.json')))
m03c = json.load(open(os.path.join(OUT, 'm03c_exact_local.json')))
m04 = json.load(open(os.path.join(OUT, 'm04_overlap.json')))
m05 = json.load(open(os.path.join(OUT, 'm05_pointprocess.json')))
d04 = json.load(open(os.path.join(D, 'd04_renull.json')))
dg = pd.read_csv(os.path.join(D, 'gap_vs_null.csv'))            # dolphin null A, 10 bins
DUR_D, DUR_M = 0.407, 2.49                                        # median call duration (s)
IOI_D, IOI_M = 0.62, 16.6                                         # median IOI: dolphin pooled chorus; marmoset per caller
LIGHT_D, LIGHT_M = DOLPHIN_LIGHT, MARMOSET_LIGHT

# ---------- dolphin gap bins ----------
dbins = [(-3, -0.5), (-0.5, -0.25), (-0.25, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 100)]
dA = dg.ratio.values
# null B (paired shuffle) — five bins reported; others ≈ 1 (the null preserves them by construction)
dB = {(-3, -0.5): 0.97, (0, 0.1): 1.01, (0.1, 0.25): 0.99, (0.25, 0.5): 1.03, (1, 2): 1.01}

# ---------- marmoset gap bins ----------
medges = [(-np.inf, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 60), (60, np.inf)]
mA = np.array(m02['tables']['all_A']['ratio']); mB = np.array(m02['tables']['all_B']['ratio']); mS = np.array(m02['tables']['other_shift']['ratio'])
fine = m02['fine']; fe = np.array(fine['edges']); fo = np.array(fine['obs']); fn = np.array(fine['null_mean']); fsd = np.array(fine['null_sd'])
fratio = fo / fn; fc = 0.5 * (fe[:-1] + fe[1:])


def step_xy(bins, vals, lo_clip, hi_clip):
    xs, ys = [], []
    for (a, b), v in zip(bins, vals):
        a = max(a, lo_clip); b = min(b, hi_clip)
        xs += [a, b]; ys += [v, v]
    return np.array(xs), np.array(ys)


f, plt = fig(FULL_W, 5.6)
gs = f.add_gridspec(1, 2, wspace=0.18, top=0.8, bottom=0.14)
# (a-left) absolute gap
ax = f.add_subplot(gs[0, 0]); style(ax, 'Absolute time', 'gap (s): next onset − latest offset', 'observed / null'); panel_label(ax, 'a', x=-0.06)
x, y = step_xy(dbins, dA, -3, 12); ax.plot(x, y, color=DOLPHIN, lw=2, label='dolphin, null A (IOI + duration shuffle)')
xb, yb = step_xy([b for b in dbins if b in dB], [dB[b] for b in dbins if b in dB], -3, 12); ax.plot(xb, yb, color=DOLPHIN, lw=2, ls=':', label='dolphin, null B (paired shuffle) — the effect vanishes')
x, y = step_xy(medges, mA, -3, 12); ax.plot(x, y, color=LIGHT_M, lw=2, label='marmoset pooled, null A (no identity)')
x, y = step_xy(medges, mB, -3, 12); ax.plot(x, y, color=MARMOSET, lw=1.6, ls=':', label='marmoset pooled, null B — flat by construction')
ax.plot(fc, fratio, color=MARMOSET, lw=2, label='marmoset other-caller vs shift null (identity)')
ax.fill_between(fc, (fn - 2 * fsd) / fn, (fn + 2 * fsd) / fn, color=MARMOSET, alpha=0.08, lw=0)
ax.axhline(1, color=INK2, lw=0.8); ax.axvline(0, color=GRID, lw=1); ax.set_xlim(-3, 12); ax.set_ylim(0, 3.1)
gap_handles, _ = ax.get_legend_handles_labels()
ax.annotate('2.06× at 3.0–3.25 s', (3.125, 2.06), xytext=(5.0, 2.55), fontsize=FS_SMALL, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.annotate('dolphin 1.26×\nat 0–0.1 s\n(null A only)', (0.05, 1.26), xytext=(-2.9, 1.75), fontsize=FS_SMALL, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.annotate('overlap avoidance 0.14–0.32×', (-1.0, 0.24), xytext=(1.3, 0.12), fontsize=FS_SMALL, color=INK2, va='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
# (a-right) rescaled: gap / median call duration
ax = f.add_subplot(gs[0, 1]); style(ax, 'Gap in call lengths', 'gap / median call duration\n(dolphin 0.41 s, marmoset 2.5 s)', None); panel_label(ax, 'b', x=-0.04)
x, y = step_xy([(a / DUR_D, b / DUR_D) for a, b in dbins], dA, -2, 6); ax.plot(x, y, color=DOLPHIN, lw=2, label='dolphin, null A')
xb, yb = step_xy([(a / DUR_D, b / DUR_D) for a, b in dbins if (a, b) in dB], [dB[b] for b in dbins if b in dB], -2, 6); ax.plot(xb, yb, color=DOLPHIN, lw=2, ls=':', label='dolphin, null B')
ax.plot(fc / DUR_M, fratio, color=MARMOSET, lw=2, label='marmoset other-caller vs shift null')
x, y = step_xy([(a / DUR_M, b / DUR_M) for a, b in medges], mA, -2, 6); ax.plot(x, y, color=LIGHT_M, lw=2, label='marmoset pooled, null A')
ax.axhline(1, color=INK2, lw=0.8); ax.axvline(0, color=GRID, lw=1); ax.set_xlim(-1.5, 5); ax.set_ylim(0, 3.1)
ax.axvspan(0, 1, color=GRID, alpha=0.5, lw=0); ax.text(0.5, 3.05, 'within one\ncall length', ha='center', va='top', fontsize=FS_SMALL, color=INK2)
f.legend(handles=gap_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=2.2, columnspacing=1.5)
headline(f, 'Turn-taking: with caller identity the coupling is unmistakable;\nwithout it, the dolphin-style nulls see little (A) or nothing (B)', y=1.03,
         sub='Dolphin: 4,595 gaps, 19 sessions, single hydrophone. Marmoset: 15,290 calls, 128 dyad/trio sessions, callers known;\nshaded band = ±2 sd of the shift null.', sub_y=0.935)
save(f, 'xs_gaps')

# ---------- (b) self-repeat ----------
dbins2 = [(0.05, 0.25), (0.25, 1), (1, 3), (3, 10)]
dr = [g['ratio_file'] for g in d04['gap']['table']]; dci = [g['ratio_file_ci'] for g in d04['gap']['table']]
dsess = [g['ratio_session'] for g in d04['gap']['table']]
mbins2 = [(5, 10), (10, 20), (20, 30), (30, 60)]
mr = m03c['ratio']; msd = np.array(m03c['sd']) / np.array(m03c['null']); mz = m03c['z']
m03 = json.load(open(os.path.join(OUT, 'm03_selfrepeat.json')))
f, plt = fig(FULL_W, 5.4)
gs = f.add_gridspec(1, 2, wspace=0.18, top=0.8, bottom=0.15)
ax = f.add_subplot(gs[0, 0]); style(ax, 'Absolute time (log)', 'gap between the two calls (s)', 'exact pairs,\nobserved / within-minute null'); panel_label(ax, 'a', x=-0.06)
gm = lambda a, b: np.sqrt(a * b)
xd = [gm(a, b) for a, b in dbins2]; xm = [gm(a, b) for a, b in mbins2]
ax.errorbar(xd, dr, yerr=[np.array(dr) - np.array(dci)[:, 0], np.array(dci)[:, 1] - np.array(dr)], fmt='o-', color=DOLPHIN, lw=2, ms=6, capsize=3, label='dolphin, within-minute file null (95% CI)')
ax.plot(xd, dsess, 'o--', color=LIGHT_D, lw=1.4, ms=4, label='dolphin, session null (drift-inflated)')
ax.errorbar(xm, mr, yerr=1.96 * msd, fmt='s-', color=MARMOSET, lw=2, ms=6, capsize=3, label='marmoset, 60-s block null (±1.96 sd)')
# marmoset session null exact-pair ratio (m03): inflated by drift
try:
    sess_exact = m03['exact']['ratio'] if 'exact' in m03 else None
except Exception:
    sess_exact = None
ax.axhline(1, color=INK2, lw=0.8); ax.set_xscale('log'); ax.set_xlim(0.03, 80); ax.set_ylim(0.8, 2.0)
ax.axvspan(0.03, 2.5, color=GRID, alpha=0.35, lw=0); ax.text(0.04, 0.83, 'marmoset refractory\nfloor ≈ 2.5 s', fontsize=FS_SMALL, color=INK2, ha='left', va='bottom')
sr_handles, _ = ax.get_legend_handles_labels()
ax = f.add_subplot(gs[0, 1]); style(ax, 'Gap in call lengths (log)', 'gap / median call duration', None); panel_label(ax, 'b', x=-0.04)
ax.errorbar(np.array(xd) / DUR_D, dr, yerr=[np.array(dr) - np.array(dci)[:, 0], np.array(dci)[:, 1] - np.array(dr)], fmt='o-', color=DOLPHIN, lw=2, ms=6, capsize=3, label='dolphin (0.41 s per call)')
ax.errorbar(np.array(xm) / DUR_M, mr, yerr=1.96 * msd, fmt='s-', color=MARMOSET, lw=2, ms=6, capsize=3, label='marmoset (2.5 s per call)')
ax.axhline(1, color=INK2, lw=0.8); ax.set_xscale('log'); ax.set_xlim(0.1, 40); ax.set_ylim(0.8, 2.0)
ax.legend(loc='upper right')
ax.text(0.03, 0.03, 'both decay to chance within ~10–20 call lengths;\nat matched gap the dolphin excess is ≈ 2× the marmoset one', transform=ax.transAxes, ha='left', fontsize=FS_SMALL, color=INK2)
f.legend(handles=sr_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=2.2, columnspacing=1.5)
headline(f, 'Self-repeat: an individual re-issues a near-identical call at short lag\nin both species — same arc, different amplitude', y=1.03,
         sub='Exact = feature distance ≤ 5th percentile of all within-session pairs. Nulls permute call identities within 60-s classes\n(marmoset) or one-minute files (dolphin), so slow drift cannot inflate the excess. Points at geometric bin centres.', sub_y=0.935)
save(f, 'xs_selfrepeat')

# ---------- (c) table ----------
pm = m05['pooled']['dyad']['fano']; pc = m05['caller']['all']['fano']
rows = [
    ('Data', 'Lehnhoff et al. 2025 — 4,637 whistles, 19 sessions, Bay of Biscay, one hydrophone', 'Grijseels et al. 2024 — 18,822 phee calls, 212 sessions, 51 captive animals, per-animal mics'),
    ('Caller identity', 'unknown (chorus of tens of animals)', 'known (1, 2 or 3 animals in separate booths)'),
    ('Call rate / median duration', '≈ 30 per min averaged over non-empty minutes, 60–400 in bursts / 0.41 s', '2.2 per animal per min / 2.5 s'),
    ('Turn-taking, best available null', 'ρ(dur, next IOI) = 0.10; 0–250 ms excess 1.26× under null A, 1.0× under null B → unidentifiable', 'other-caller latency 2.06× at 3.0–3.25 s vs shift null; 87/128 sessions; null B = 1.00 (blind by construction)'),
    ('Response window', '— (0–250 ms if real)', '1–5 s, peak ≈ 3 s; ≈ 1.2 call lengths after the partner ends'),
    ('Overlap vs null', '0.87–0.97× deep overlaps (z −1.6 under null B); no jamming avoidance', f'{m04["overlap"]["ratio_shift"]:.2f}× vs shift null, 109/128 sessions; collisions (<0.5 s) 0.78×, later interruptions 0.16×'),
    ('Same-caller refractory', 'unmeasurable', '≈ 2.5 s hard floor; 4 gaps < 2 s in 7,288 dyad transitions'),
    ('Exact self-repeat, session null', '1.46× (≤ 10 s)', '2.4× (5–10 s) — both inflated by slow drift'),
    ('Exact self-repeat, local null', '1.44× (< 0.25 s) → 1.13× (3–10 s); participation +3.5 pct pts', f'{mr[0]:.2f}× (5–10 s) → {mr[3]:.2f}× (30–60 s); participation +{m03c["participation"]["excess_pct"]:.1f} pct pts'),
    ('What carries the repeat', 'absolute pitch (shape-matched, pitch-shifted pairs depleted)', 'f_max ratio 0.944 — frequency strongest of 7 features'),
    ('Cross-caller matching', 'none (pitch-shifted copies depleted)', 'none: f ratios ≈ 0.99 n.s.; loudness 0.88 (mic confound)'),
    ('Fano factor 1 s / 10 s / 60 s', '1.43 / 4.06 / 10.9 (renewal 1.29 / 2.23 / 2.63)', f'{pm["1"]["real"]:.2f} / {pm["10"]["real"]:.2f} / {pm["60"]["real"]:.2f} (renewal {pm["1"]["renewal"]:.2f} / {pm["10"]["renewal"]:.2f} / {pm["60"]["renewal"]:.2f}), dyad pooled'),
    ('Burstiness B', '+0.42 (bursty)', f'{m05["pooled"]["all"]["B_mean"]:+.2f} pooled, {m05["caller"]["all"]["B_mean"]:+.2f} per caller (regular)'),
    ('Process', 'Cox: drifting rate × renewal; Hawkes unidentifiable', 'refractory renewal × slow drift; Hawkes α ≈ 0 in 44/75 sessions; where α > 0 the fitted memory is 45–1000 s (drift, not call-triggers-call)'),
]
# Layout in inches so the table keeps the shared type scale at the 10-in figure width. Row heights follow the wrapped text.
BODY = FS_SMALL; KEYF = FS_SMALL + 0.5; LINE = BODY * 1.35 / 72          # line height (in)
W = FULL_W; col_in = [0.0, 2.25, 6.05]; wrap_key, wrap_a, wrap_b = 26, 46, 52
wrapped = [(textwrap.wrap(k, wrap_key), textwrap.wrap(a, wrap_a), textwrap.wrap(b, wrap_b)) for k, a, b in rows]
row_h = [max(len(x) for x in r) * LINE + 0.12 for r in wrapped]
head_h = 0.45; H = head_h + sum(row_h) + 0.1
f, plt = fig(W, H)
ax = f.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.text(col_in[1], 0.3, 'Common dolphin chorus\n(DOLPHINFREE)', fontsize=FS_LEGEND, color=DOLPHIN, weight='bold', va='bottom')
ax.text(col_in[2], 0.3, 'Marmoset dyads/trios\n(Grijseels 2024)', fontsize=FS_LEGEND, color=MARMOSET, weight='bold', va='bottom')
y = head_h
for i, ((k, a, b), h) in enumerate(zip(wrapped, row_h)):
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((-0.05, y), W + 0.1, h, color=GRID, alpha=0.45, lw=0))
    for x, lines, fs, c, wt in [(col_in[0] + 0.05, k, KEYF, INK, 'bold'), (col_in[1], a, BODY, INK2, 'normal'), (col_in[2], b, BODY, INK2, 'normal')]:
        ax.text(x, y + 0.07, '\n'.join(lines), fontsize=fs, color=c, weight=wt, va='top', linespacing=1.35)
    y += h
headline(f, 'Cross-species comparison — every number against a null;\nsame statistical skeleton, different strategies', y=1.0 + 0.75 / H)
save(f, 'xs_table')
print('done')
