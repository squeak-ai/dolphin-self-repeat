"""m07 — Null B power figure (supplementary; not numbered in the main article).

The same marmoset calls, the same gap histogram, three nulls:
  shift null   — caller identity known; one caller's call train is circularly shifted against the other's.
  Null A       — identity discarded, calls pooled as one chorus; IOIs and durations shuffled independently.
  Null B       — identity discarded, pooled; (duration, next IOI) pairs shuffled together.
The shift null finds overlap avoidance and a response peak at z = +18.5. Null A sees a faint echo. Null B returns
~1.00 in every bin: it preserves every gap by construction, so it cannot differ from the data.

Numbers are read from m02_gaps.json (tables other_shift, all_A, all_B) — the source of the article's Null B table.
Output: null_b_power.{png,pdf}
"""
import sys
import numpy as np
from m00_lib import OUT, FIG, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from figstyle import *

m02 = json.load(open(os.path.join(OUT, 'm02_gaps.json')))
T = m02['tables']
KEEP = ['-0.5–0', '0–0.5', '0.5–1', '1–2', '2–3', '3–5', '5–10']
LABELS = ['−0.5–0\n(overlap)', '0–0.5', '0.5–1', '1–2', '2–3', '3–5', '5–10']
series = [('other_shift', MARMOSET, 'shift null — caller identity known'),
          ('all_A', MARMOSET_LIGHT, 'Null A — identity discarded, independent shuffle'),
          ('all_B', G2, 'Null B — identity discarded, paired shuffle')]


def pick(tab, key):
    return np.array([tab[key][tab['bin'].index(b)] for b in KEEP], float)


f, ax = plt.subplots(figsize=(FULL_W, 5.6))
style(ax, xlabel='gap from the end of one call to the onset of the next (s)', ylabel='observed / null')
x = np.arange(len(KEEP)); w = 0.27
ax.axhline(1, color=INK2, lw=0.9, zorder=1)
for i, (k, c, lab) in enumerate(series):
    tab = T[k]; r = pick(tab, 'ratio'); obs = pick(tab, 'obs'); nul = pick(tab, 'null'); sd = pick(tab, 'sd')
    err = 2 * sd * obs / nul ** 2                          # ±2 sd of the null, propagated to the ratio
    xi = x + (i - 1) * w
    ax.bar(xi, r - 1, w - 0.03, bottom=1, color=c, edgecolor=SURF, lw=0.8, zorder=2, label=lab)
    ax.errorbar(xi, r, yerr=err, fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    if k == 'other_shift':
        z = pick(tab, 'z')
        for xx, rr, zz, ee in zip(xi, r, z, err):
            lab_z = f'z = {zz:+.1f}'.replace('-', '−')
            if rr >= 1:
                ax.text(xx, rr + ee + 0.03, lab_z, ha='center', va='bottom', fontsize=FS_SMALL, color=INK2)
            elif rr < 0.6:                                   # long bar: label inside, reading downward
                ax.text(xx, 0.975, lab_z, ha='center', va='top', fontsize=FS_SMALL, color=SURF, rotation=90)
            else:                                            # short bar: label below its error bar
                ax.text(xx, rr - ee - 0.03, lab_z, ha='center', va='top', fontsize=FS_SMALL, color=INK2)
    if k == 'all_B':
        rb, zb = r, pick(tab, 'z')
        ax.plot(xi, r, color=G3, lw=1.6, marker='D', ms=5, zorder=4)    # make the flatness itself visible

ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_xlim(-0.55, len(KEEP) - 0.45); ax.set_ylim(0, 2.1)
ax.legend(loc='upper left', handlelength=1.4)
ax.annotate(f'Null B: {rb.min():.2f}–{rb.max():.2f} in every bin\nthe instrument sees nothing',
            (x[5] + w, rb[5] - 0.02), xytext=(x[5] + 0.45, 0.45), fontsize=FS_ANNO, color=INK, ha='center',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
headline(f, 'Same calls, three nulls: the shift null finds z = +18.5; Null B finds nothing', y=1.02,
         sub='Marmoset dyads and trios (Grijseels et al. 2024), 15,290 calls, 128 sessions. Error bars: ±2 sd of each null.\n'
             'Null B keeps every (duration, next interval) pair — and therefore every gap — so it reproduces the data by construction.',
         sub_y=0.975)
f.subplots_adjust(top=0.86)
save_both(f, FIG, 'null_b_power')
