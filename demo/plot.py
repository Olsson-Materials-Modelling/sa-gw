#!/usr/bin/env python3
"""The figures, one row each: the pipeline on one cell (1-4) and the example classes (5-11); demo/out/*.pdf|png, demo/figures/*.png."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo')); sys.path.insert(0, str(ROOT / 'example'))
from sagw.structures import read_structure_files, distances, minimum_image, displacement
from sagw.graph import contact_graph, graph_weights, operator
from sagw.network import Network
from sagw.spin import shares
from style import INK, GREY, TWO_COLUMN, apply, style_axes, legend, panel_label, save, classes_of, palette, two_line
from make_example import GAP

OUT, FIG = ROOT / 'demo/out', ROOT / 'demo/figures'
EDGE = dict(edgecolors=INK, linewidths=0.4); SHORT = ['deficit', 'excess', 'bond mean', 'bond spread', 'outer rank', 'next gap', 'angle', 'site potential', 'confinement', 'share', 'log share']
ROW = dict(figsize=(TWO_COLUMN, 2.9), constrained_layout=True)


def points(ax, x, y, c, COL, MARK, **kw):
    """One class as filled markers with an ink edge."""
    return ax.scatter(x, y, s=22, c=COL[c], marker=MARK[c], label=c, **EDGE, **kw)


def project(pos, cell, t, m):
    """Positions of the atoms in m relative to atom t on their two principal axes, and the projection."""
    r = minimum_image(pos[m] - pos[t], cell); _, _, vt = np.linalg.svd(r - r.mean(0), full_matrices=False); return r @ vt[:2].T, vt[:2].T


def neighbourhood(ax, pos, cell, sym, hop, t, colour, net):
    """The trap atom and the atoms within two bonds: neutral contacts in grey, the relaxation as arrows."""
    m = hop <= 2; p0, P = project(pos[0], cell, t, m); p1 = p0 + minimum_image(pos[1, m] - pos[0, m], cell) @ P
    role = net.role_of(sym[m]); c = contact_graph(distances(pos[0, m], cell), role, net); k = int(np.flatnonzero(np.flatnonzero(m) == t)[0])
    for i, j in zip(*np.nonzero(np.triu(c))): ax.plot([p0[i, 0], p0[j, 0]], [p0[i, 1], p0[j, 1]], '-', color='.8', lw=1, zorder=1)
    ax.scatter(p0[role, 0], p0[role, 1], s=24, c='.5', zorder=2, **EDGE); ax.scatter(p0[~role, 0], p0[~role, 1], s=14, c='w', zorder=2, **EDGE)
    for i in range(len(p0)):
        if np.linalg.norm(p1[i] - p0[i]) > .05: ax.annotate('', p1[i], p0[i], arrowprops=dict(arrowstyle='->', color=colour, lw=1.2), zorder=3)
    ax.scatter([p0[k, 0]], [p0[k, 1]], s=80, c=colour, zorder=4, **EDGE); ax.set_aspect('equal', adjustable='datalim'); ax.set_xticks([]); ax.set_yticks([]); style_axes(ax, False, False)


def heat(ax, M, cmap, vmin, vmax, rows, cols, fs=8):
    """A small labelled matrix."""
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto'); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=fs); ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=fs, rotation=90); style_axes(ax, False, False); ax.tick_params(which='both', length=0); return im


def letters(axes, xs):
    for a, letter, x in zip(axes, 'abcdefgh', xs): panel_label(a, letter, x=x)


def main():
    apply(); FIG.mkdir(exist_ok=True); net = Network.load(ROOT / 'example/data/network.json'); SCALES = net.scales; scale_lab = [f'{t:g}' for t in SCALES]
    ev = pd.read_csv(OUT / 'events.csv'); z = np.load(OUT / 'fingerprint.npz', allow_pickle=True); S = z['S'].astype(float); Z = z['Z'].astype(float); hop = z['hop']; trap = z['trap']
    sep = pd.read_csv(OUT / 'class_separation.csv'); model = pd.read_csv(OUT / 'model.csv'); ident = pd.read_csv(OUT / 'identification.csv'); conf = pd.read_csv(OUT / 'confusion.csv'); sig = pd.read_csv(OUT / 'response_signature.csv'); rad = pd.read_csv(OUT / 'scale_radius.csv')
    boxes, sym, cell, pos = read_structure_files(ROOT / 'example/data/structures'); CLASSES = classes_of(ev.kind); COL, MARK = palette(CLASSES); nk = len(CLASSES); TWO_LINE = [two_line(c) for c in CLASSES]; K = {c: (ev.kind == c).to_numpy() for c in CLASSES}
    rep = {c: int(ev.index[K[c]][np.argsort(ev.eta__structure[K[c]].to_numpy())[K[c].sum() // 2]]) for c in CLASSES}     # the median-eta cell of each class
    shown = 'bond break' if 'bond break' in rep else CLASSES[0]; b = rep[shown]; t = int(trap[b]); role = net.role_of(sym[b]); d = [distances(pos[b, q], cell[b]) for q in range(2)]; con = [contact_graph(d[q], role, net) for q in range(2)]; w = graph_weights(d[0], con[0], net); H = operator(w); hb = hop[b]; blue = COL[shown]
    # 1 the contact graph, the channels and the operator of one bond-break cell
    fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.1, 1, 1]), **ROW)
    m = hb <= 3; p, _ = project(pos[b, 0], cell[b], t, m); idx = np.flatnonzero(m); k = int(np.flatnonzero(idx == t)[0]); wm = w[np.ix_(idx, idx)]; lost = con[0] & ~con[1]
    for i, j in zip(*np.nonzero(np.triu(wm))): ax[0].plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], '--' if lost[idx[i], idx[j]] else '-', color=blue if lost[idx[i], idx[j]] else '.75', lw=.4 + 2.2 * wm[i, j] / w.max(), zorder=1)
    ax[0].scatter(p[role[idx], 0], p[role[idx], 1], s=22, c='.5', zorder=2, **EDGE); ax[0].scatter(p[~role[idx], 0], p[~role[idx], 1], s=12, c='w', zorder=2, **EDGE); ax[0].scatter([p[k, 0]], [p[k, 1]], s=70, c=blue, zorder=3, **EDGE)
    ax[0].set_aspect('equal', adjustable='datalim'); ax[0].set_xticks([]); ax[0].set_yticks([]); style_axes(ax[0], False, False); ax[0].set_xlabel(f'contacts within {net.cutoff:.2f} Å, width = weight\ndashed: lost on relaxation', fontsize=8)
    order = np.argsort(hb[m], kind='stable'); rows = idx[order][:12]; heat(ax[1], Z[b, 0, rows, :7], 'RdBu_r', -3, 3, [f'{hb[i]}' for i in rows], SHORT[:7]); ax[1].set_ylabel('atoms, by bonds from the trap atom', fontsize=8); ax[1].set_xlabel('channels x, standardised', fontsize=8)
    e = np.zeros((len(hb), 1)); e[t] = 1
    for tt, r_ in zip(SCALES, rad.radius_median_A): h = H(tt, e)[:, 0]; ax[2].plot(range(6), [h[hb == i].mean() for i in range(6)], marker='o', ms=3.5, mec=INK, mew=.3, lw=1.2, label=f'{tt:g} ({r_:.1f} Å)')
    ax[2].set_yscale('log'); ax[2].set_xticks(range(6)); ax[2].set_xlabel('bonds from the trap atom'); ax[2].set_ylabel(r'$H_t$ from the trap atom'); style_axes(ax[2], minor_x=False); legend(ax[2], fontsize=7, title='scale t (reach)', title_fontsize=7, loc='lower left')
    letters(ax, (-0.02, -0.2, -0.26)); save(fig, OUT, FIG, 'fig01_graph_channels_operator')
    # 2 the fingerprint at the trap atom and its readings
    V0, V1 = S[b, 0, t, :, :7], S[b, 1, t, :, :7]; vmax = max(V0.max(), V1.max()); fig, ax = plt.subplots(1, 4, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]), **ROW)
    for i, (M, lab, cmap, lo, hi) in enumerate(((V0, r'neutral $V_0$', 'Blues', 0, vmax), (V1, r'relaxed anion $V^-$', 'Blues', 0, vmax), (V1 - V0, r'response $V^- - V_0$', 'RdBu_r', -vmax, vmax))):
        heat(ax[i], M, cmap, lo, hi, scale_lab if i == 0 else [''] * 5, SHORT[:7]); ax[i].set_xlabel(lab, fontsize=9)
    ax[0].set_ylabel('scale t'); r0, r1 = np.linalg.norm(V0, axis=1), np.linalg.norm(V1, axis=1); et = np.linalg.norm(V1 - V0, axis=1) / r0
    ax[3].plot(SCALES, r0, marker='o', ms=4, mec=INK, mew=.4, color=GREY, label=r'$\|V_{0,t}\|$'); ax[3].plot(SCALES, r1, marker='s', ms=4, mec=INK, mew=.4, color=INK, label=r'$\|V^-_t\|$'); ax[3].plot(SCALES, et, marker='D', ms=4, mec=INK, mew=.4, color=blue, label=r'$\eta_t$')
    ax[3].set_xscale('log'); ax[3].set_xticks(SCALES); ax[3].set_xticklabels(scale_lab); ax[3].minorticks_off(); ax[3].set_xlabel('scale t'); ax[3].set_ylabel('norm at the trap atom'); style_axes(ax[3]); legend(ax[3], fontsize=7.5, loc='upper right')
    ax[3].text(.04, .04, f'$T_0$ = {r0.sum():.2f}\n$\\eta$ = {np.linalg.norm(V1 - V0) / np.linalg.norm(V0):.2f}', transform=ax[3].transAxes, fontsize=8, va='bottom', bbox=dict(boxstyle='square,pad=.3', fc='w', ec=INK, lw=.6))
    letters(ax, (-0.3, -0.08, -0.08, -0.3)); save(fig, OUT, FIG, 'fig02_fingerprint_readings')
    # 3 the potential and electron families and the full response
    fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.1]), **ROW); a = np.load(ROOT / 'example/data/potentials.npz'); site, contrast = a['site_eV'][b], a['shell_eV'][b] - a['site_eV'][b]
    ax[0].scatter(site[0], contrast[0], s=12, facecolors='none', edgecolors='.55', linewidths=.6, label='every atom, neutral'); ax[0].scatter(site[1], contrast[1], s=12, c='.55', label='every atom, relaxed anion', **EDGE)
    ax[0].scatter([site[0, t]], [contrast[0, t]], s=60, facecolors='none', edgecolors=blue, linewidths=1.2, zorder=3, label='trap atom, neutral'); ax[0].scatter([site[1, t]], [contrast[1, t]], s=60, c=blue, zorder=3, label='trap atom, relaxed', **EDGE)
    ax[0].set_xlabel('potential at the site (eV)'); ax[0].set_ylabel('confinement (eV)'); style_axes(ax[0]); legend(ax[0], fontsize=6.5, loc='upper right', handletextpad=.3)
    el = pd.read_parquet(ROOT / 'example/data/electron.parquet'); el = el[el.cell == boxes[b]]; x = np.arange(7)
    for i, (st, lab, col) in enumerate((('P', 'before trapping', GREY), ('T', 'after relaxation', blue))):
        s_ = el[el.state == st].sort_values('atom').share.to_numpy(); n_eff = shares(s_)[1]; ax[1].bar(x + (i - .5) * .38, [s_[hb == h].sum() for h in range(7)], .38, color=col, edgecolor=INK, lw=.5, label=f'{lab} ($N_\\mathrm{{eff}}$ {n_eff:.0f})')
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(h) for h in range(6)] + ['6+']); ax[1].set_xlabel('bonds from the trap atom'); ax[1].set_ylabel('share of the added electron'); style_axes(ax[1], minor_x=False); legend(ax[1], fontsize=7, loc='upper right')
    R = S[b, 1, t] - S[b, 0, t]; v = np.abs(R).max(); heat(ax[2], R, 'RdBu_r', -v, v, scale_lab, SHORT, fs=7.5)
    for j in (7, 9): ax[2].axvline(j - .5, color=INK, lw=.8)
    ax[2].set_ylabel('scale t'); ax[2].set_xlabel('full response at the trap atom', fontsize=9); letters(ax, (-0.28, -0.26, -0.2)); save(fig, OUT, FIG, 'fig03_families')
    # 4 the displacement field, its wavelets and the descriptors
    fig, ax = plt.subplots(1, 3, **ROW); u = displacement(pos[b, 0], pos[b, 1], cell[b]); mag = np.linalg.norm(u, axis=1)
    ax[0].scatter(hb + np.random.default_rng(0).uniform(-.18, .18, len(hb)), mag, s=10, c='.55', **EDGE); ax[0].scatter([0], [mag[t]], s=50, c=blue, zorder=3, **EDGE); ax[0].set_yscale('log'); ax[0].set_xticks(range(7)); ax[0].set_xlabel('bonds from the trap atom'); ax[0].set_ylabel('displacement of the atom (Å)'); style_axes(ax[0], minor_x=False)
    pw = [ev.loc[b, f'displacement_power_t{tt:g}_A2'] for tt in SCALES]; ax[1].bar(range(len(SCALES)), pw, color=blue, edgecolor=INK, lw=.5); ax[1].set_xticks(range(len(SCALES))); ax[1].set_xticklabels(scale_lab); ax[1].set_xlabel('scale t'); ax[1].set_ylabel('wavelet power (Å$^2$)')
    ax[1].text(.96, .96, f'$Q_D$ = {ev.graph_deformation_A2[b]:.2f} Å$^2$', transform=ax[1].transAxes, ha='right', va='top', fontsize=8, bbox=dict(boxstyle='square,pad=.3', fc='w', ec=INK, lw=.6)); style_axes(ax[1], minor_x=False)
    X = pd.read_csv(OUT / 'event_descriptors.csv'); groups = {'response': X.columns.str.startswith('response__'), 'wavelet': X.columns.str.startswith('wavelet__'), 'quadratic': X.columns.str.startswith('quadratic__')}
    ax[2].bar(range(3), [int(g.sum()) for g in groups.values()], color=[GREY, blue, '.3'], edgecolor=INK, lw=.5); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(list(groups)); ax[2].set_ylabel(f'descriptors of the event ({X.shape[1]})'); ax[2].set_xlabel('descriptor block'); style_axes(ax[2], minor_x=False)
    letters(ax, (-0.3, -0.3, -0.3)); save(fig, OUT, FIG, 'fig04_displacement_descriptors')
    # 5 the five classes
    fig, ax = plt.subplots(1, nk, figsize=(TWO_COLUMN, 1.1 + 5.5 / nk), constrained_layout=True)
    for a_, c in zip(ax, CLASSES):
        bb = rep[c]; neighbourhood(a_, pos[bb], cell[bb], sym[bb], hop[bb], int(trap[bb]), COL[c], net); a_.set_xlabel(f'{c}\n{ev.contacts_before[bb]} → {ev.contacts_after[bb]} contacts', fontsize=9)
    handles = [Line2D([], [], ls='', marker='o', ms=5, mfc='.5', mec=INK, mew=.4), Line2D([], [], ls='', marker='o', ms=4, mfc='w', mec=INK, mew=.4), Line2D([], [], ls='', marker='o', ms=7, mfc=GREY, mec=INK, mew=.4), Line2D([], [], color=INK, lw=1.2)]
    leg = fig.legend(handles, ['A (four-fold)', 'B (two-fold)', 'trap atom', 'relaxation'], loc='outside upper center', ncol=4, fontsize=8, handletextpad=.4, columnspacing=1.2); leg.get_frame().set_linewidth(.8)
    letters(ax, (0.0,) * nk); save(fig, OUT, FIG, 'fig05_classes')
    # 6 the physics each class carries
    fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.5, 1, 1]), figsize=(TWO_COLUMN, 3.1), constrained_layout=True)
    for i, c in enumerate(CLASSES):
        k = K[c]; jit = (np.arange(k.sum()) - k.sum() / 2) * .012; mb, ma = ev.level_before_eV[k].median(), ev.level_after_eV[k].median()
        ax[0].scatter(ev.level_before_eV[k], i - .2 + jit, s=12, facecolors='none', edgecolors=COL[c], linewidths=.7, zorder=2); ax[0].scatter(ev.level_after_eV[k], i + .2 + jit, s=12, c=COL[c], marker=MARK[c], zorder=2, **EDGE)
        ax[0].annotate('', (ma, i), (mb, i), arrowprops=dict(arrowstyle='-|>', color=COL[c], lw=1.8, shrinkA=0, shrinkB=0), zorder=3)
        points(ax[1], ev.n_eff_before[k], ev.n_eff_after[k], c, COL, MARK); points(ax[2], ev.site_potential_before_eV[k], ev.site_potential_after_eV[k], c, COL, MARK)
    for x, ls in ((0, '-'), (-GAP / 2, ':'), (-GAP, '-')): ax[0].axvline(x, color=GREY, lw=.8, ls=ls, zorder=1)
    for x, lab, ha in ((-GAP, 'valence band', 'center'), (0, 'conduction band', 'right')): ax[0].text(x, -.68, lab, ha=ha, va='bottom', fontsize=7.5, color=GREY, clip_on=False)
    ax[0].set_yticks(range(nk)); ax[0].set_yticklabels(CLASSES, fontsize=8); ax[0].set_ylim(nk - .4, -.6); ax[0].set_xlim(-GAP - 1.3, .5); ax[0].set_xlabel('level (eV below the conduction band edge)\nopen: empty, before; filled: occupied, after', fontsize=8); style_axes(ax[0], minor_y=False); ax[0].tick_params(axis='y', which='both', length=0)
    lim = [.8, 150]; ax[1].plot(lim, lim, ':', color=GREY, lw=1); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel('atoms carrying the electron\nbefore trapping', fontsize=8); ax[1].set_ylabel('atoms carrying the electron\nafter relaxation', fontsize=8); style_axes(ax[1]); legend(ax[1], loc='upper left', fontsize=6.5, handletextpad=.3)
    lim = [-1.7, .1]; ax[2].plot(lim, lim, ':', color=GREY, lw=1); ax[2].set_xlim(lim); ax[2].set_ylim(lim); ax[2].set_xlabel('potential at the trap atom\nbefore trapping (eV)', fontsize=8); ax[2].set_ylabel('potential at the trap atom\nafter relaxation (eV)', fontsize=8); style_axes(ax[2])
    letters(ax, (-0.36, -0.42, -0.42)); save(fig, OUT, FIG, 'fig06_physics')
    # 7 the fingerprint plane and the change by scale
    fig, ax = plt.subplots(1, 2, **ROW)
    for c in CLASSES:
        k = K[c]; points(ax[0], ev.T0__structure[k], ev.eta__structure[k], c, COL, MARK)
        prof = ev.loc[k, [f'eta_t{tt:g}__structure' for tt in SCALES]].to_numpy(); med = np.median(prof, 0); ax[1].errorbar(SCALES, med, yerr=[med - np.quantile(prof, .25, 0), np.quantile(prof, .75, 0) - med], color=COL[c], marker=MARK[c], ms=4.5, mec=INK, mew=.4, capsize=2, lw=1.3, label=c)
    ax[0].set_xlabel(r'$T_0$, neutral fingerprint amplitude'); ax[0].set_ylabel(r'$\eta$, change on relaxation'); style_axes(ax[0]); legend(ax[0], loc='upper right', fontsize=7.5, handletextpad=.3)
    ax[1].set_xscale('log'); ax[1].set_xlabel('scale t (median reach, Å)'); ax[1].set_ylabel(r'$\eta_t$ (median, quartiles)'); ax[1].set_xticks(SCALES); ax[1].set_xticklabels([f'{tt:g}\n({r_:.1f})' for tt, r_ in zip(SCALES, rad.radius_median_A)]); ax[1].minorticks_off(); style_axes(ax[1])
    letters(ax, (-0.2, -0.2)); save(fig, OUT, FIG, 'fig07_plane_scales')
    # 8 the reach by bonds
    fig, ax = plt.subplots(1, 2, **ROW); hh = np.arange(6)
    for c in CLASSES:
        k = K[c]; ax[0].plot(hh, np.nanmedian(ev.loc[k, [f'reach_h{i}__structure' for i in hh]].to_numpy(), 0), color=COL[c], marker=MARK[c], mec=INK, mew=.4, lw=1.3, label=c); ax[1].plot(hh, np.nanmedian(ev.loc[k, [f'shell_displacement_h{i}_A' for i in hh]].to_numpy(), 0), color=COL[c], marker=MARK[c], mec=INK, mew=.4, lw=1.3, label=c)
    ax[0].set_xlabel('bonds from the trap atom'); ax[0].set_ylabel(r'$|S^- - S_0|$ (median)'); ax[0].set_yscale('log'); ax[1].set_xlabel('bonds from the trap atom'); ax[1].set_ylabel('displacement (Å, median)'); ax[1].set_yscale('log')
    for a_ in ax: a_.set_xticks(hh); style_axes(a_, minor_x=False)
    legend(ax[0], loc='upper right', fontsize=7.5, handletextpad=.3); letters(ax, (-0.2, -0.2)); save(fig, OUT, FIG, 'fig08_reach')
    # 9 identification held out and the confusion matrix
    fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.1, 1]), figsize=(TWO_COLUMN, 3.0), constrained_layout=True)
    names = {'contacts and deformation': 'contacts and\ndeformation', 'neutral fingerprint V0 (structure)': 'neutral fingerprint\n$V_0$, structure', 'response V- - V0 (structure)': 'response\n$V^- - V_0$, structure', 'neutral and response (structure)': 'neutral and response,\nstructure', 'full fingerprint (all families)': 'full fingerprint,\nall families'}
    y = np.arange(len(ident)); ax[0].barh(y, ident.accuracy, xerr=ident.accuracy_sd, color=[GREY] + [blue] * (len(ident) - 1), edgecolor=INK, lw=.6, height=.62, error_kw=dict(lw=.8, capsize=2)); ax[0].axvline(ident.chance[0], ls=':', color=INK, lw=.8); ax[0].text(ident.chance[0] + .02, -.75, 'chance', fontsize=8, va='center')
    ax[0].set_yticks(y); ax[0].set_yticklabels([f'{names[f]} ({n})' for f, n in zip(ident.features, ident.n_features)], fontsize=8); ax[0].set_xlim(0, 1.02); ax[0].set_ylim(len(ident) - .4, -1.1); ax[0].set_xlabel('held-out accuracy'); style_axes(ax[0], minor_y=False); ax[0].tick_params(axis='y', which='both', length=0)
    cm = conf[conf.features == 'full fingerprint (all families)'].pivot(index='true', columns='predicted', values='count').loc[CLASSES, CLASSES].to_numpy(); ax[1].imshow(cm, cmap='Blues', vmin=0, vmax=cm.max(), aspect='auto')
    for i in range(nk):
        for j in range(nk): ax[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='w' if cm[i, j] > cm.max() / 2 else INK)
    ax[1].set_xticks(range(nk)); ax[1].set_xticklabels(TWO_LINE, fontsize=7); ax[1].set_yticks(range(nk)); ax[1].set_yticklabels(TWO_LINE, fontsize=7); ax[1].set_xlabel('predicted class, full fingerprint'); ax[1].set_ylabel('true class'); style_axes(ax[1], False, False); ax[1].tick_params(which='both', length=0)
    letters(ax, (-0.62, -0.3)); save(fig, OUT, FIG, 'fig09_identification')
    # 10 the pairwise separation of every reading
    order = ['T0', 'eta', 'neutral fingerprint V0', 'response V- - V0', 'full fingerprint (all families)']; rows = [r'$T_0$', r'$\eta$', r'$V_0$, structure', r'$V^- - V_0$, structure', 'full fingerprint']
    piv = sep.pivot(index='reading', columns='classes', values='auc').loc[order].T; fig, ax = plt.subplots(figsize=(5.2, 1.3 + .31 * len(piv)), constrained_layout=True); ax.imshow(piv.to_numpy(), vmin=.5, vmax=1, cmap='Blues', aspect='auto')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f'{piv.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8, color='w' if piv.iloc[i, j] > .8 else INK)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(list(piv.index), fontsize=8); ax.set_xticks(range(len(order))); ax.set_xticklabels([r'$T_0$', r'$\eta$', r'$V_0$' + '\nstructure', r'$V^- - V_0$' + '\nstructure', 'full\nfingerprint'], fontsize=8); ax.set_xlabel('reading (AUC, held out)'); ax.set_ylabel('pair of classes'); style_axes(ax, False, False); ax.tick_params(which='both', length=0)
    save(fig, OUT, FIG, 'fig10_separation')
    # 11 the energetics and the response signature of each class
    fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.5]), figsize=(TWO_COLUMN, 3.0), constrained_layout=True)
    for c in CLASSES:
        k = K[c]; points(ax[0], ev.graph_deformation_A2[k], ev.stabilisation_eV[k], c, COL, MARK); points(ax[1], ev.stabilisation_eV[k], ev.stabilisation_predicted_eV[k], c, COL, MARK)
    ax[0].set_xlabel(r'graph deformation $Q_D$ (Å$^2$)'); ax[0].set_ylabel('stabilisation (eV)'); style_axes(ax[0]); legend(ax[0], loc='center right', fontsize=7, handletextpad=.3)
    lim = [ev.stabilisation_eV.min() - .1, ev.stabilisation_eV.max() + .1]; ax[1].plot(lim, lim, ':', color=GREY, lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel('stabilisation (eV)'); ax[1].set_ylabel('held-out prediction (eV)'); style_axes(ax[1])
    M = np.stack([sig[sig.kind == c].response.to_numpy() for c in CLASSES]); nc = M.shape[1] // len(SCALES); v = np.abs(M).max(); im = ax[2].imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for k in range(len(SCALES)):
        if k: ax[2].axvline(nc * k - .5, color=INK, lw=.8)
        for j in (7, 9): ax[2].axvline(nc * k + j - .5, color='w', lw=.6)                                               # structure | potential | electron
    ax[2].set_yticks(range(nk)); ax[2].set_yticklabels(CLASSES, fontsize=8); ax[2].set_xticks(nc * np.arange(len(SCALES)) + nc / 2 - .5); ax[2].set_xticklabels(scale_lab); ax[2].set_xlabel('scale t\n(each block: structure | potential | electron)', fontsize=8); style_axes(ax[2], False, False); ax[2].tick_params(which='both', length=0)
    cb = fig.colorbar(im, ax=ax[2], fraction=.04, pad=.03); cb.set_label('median response (standardised)', fontsize=7.5); cb.ax.tick_params(labelsize=7, direction='in'); cb.outline.set_linewidth(.8)
    letters(ax, (-0.26, -0.26, -0.34)); save(fig, OUT, FIG, 'fig11_energetics_signature')


if __name__ == '__main__':
    main()
