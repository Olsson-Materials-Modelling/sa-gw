#!/usr/bin/env python3
"""The workflow schematic (demo/figures/fig00_workflow.png): every stage drawn from one example cell, labelled with its library call."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo'))
from sagw.structures import read_structure_files, distances
from sagw.graph import contact_graph, graph_weights, operator
from sagw.network import Network
from style import INK, GREY, TWO_COLUMN, apply, save, classes_of, palette
import plot

OUT, FIG = ROOT / 'demo/out', ROOT / 'demo/figures'


def frame(ax):
    """A boxed thumbnail without ticks."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(.8)


def label(fig, x, y, title, call):
    """Stage name and library call under a thumbnail."""
    fig.text(x, y, title, ha='center', va='top', fontsize=7.5); fig.text(x, y - .045 * (1 + title.count('\n')) - .012, call, ha='center', va='top', fontsize=6, family='monospace', color='.3')


def arrow(fig, a, b):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9, color=INK, lw=.8, shrinkA=0, shrinkB=0))


def main():
    apply(); net = Network.load(ROOT / 'example/data/network.json'); ev = pd.read_csv(OUT / 'events.csv'); z = np.load(OUT / 'fingerprint.npz', allow_pickle=True); S = z['S'].astype(float); Z = z['Z'].astype(float); hop = z['hop']; trap = z['trap']
    boxes, sym, cell, pos = read_structure_files(ROOT / 'example/data/structures'); CLASSES = classes_of(ev.kind); COL, MARK = palette(CLASSES); K = {c: (ev.kind == c).to_numpy() for c in CLASSES}
    shown = 'bond break' if 'bond break' in CLASSES else CLASSES[0]; k = ev.index[K[shown]]; b = int(k[np.argsort(ev.eta__structure[k].to_numpy())[len(k) // 2]]); t = int(trap[b]); hb = hop[b]; role = net.role_of(sym[b]); blue = COL[shown]
    d = [distances(pos[b, q], cell[b]) for q in range(2)]; con = [contact_graph(d[q], role, net) for q in range(2)]; w = graph_weights(d[0], con[0], net); H = operator(w)
    fig = plt.figure(figsize=(TWO_COLUMN, 4.2)); W, Hh = .12, .29; xs = [.025 + i * .165 for i in range(6)]; y1 = .65
    # top row: inputs, settings, contact graph, channels, operator, scattering
    ax = fig.add_axes([xs[0], y1, W, Hh]); plot.neighbourhood(ax, pos[b], cell[b], sym[b], hb, t, blue, net); frame(ax)
    ax = fig.add_axes([xs[1], y1, W, Hh]); ax.axis('off'); ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='square,pad=0', fc='w', ec=INK, lw=.8, transform=ax.transAxes))
    ax.text(.08, .92, f"role_a   A X I\nexpected A4 B2\n         X4 I2\nbond     {net.bond:.2f} Å\ncutoff   {net.cutoff:.2f} Å\nsigma    {net.sigma:.2f} Å\nscales   0.5-8", transform=ax.transAxes, va='top', fontsize=6.3, family='monospace')
    ax = fig.add_axes([xs[2], y1, W, Hh]); m = hb <= 3; p, _ = plot.project(pos[b, 0], cell[b], t, m); idx = np.flatnonzero(m); kk = int(np.flatnonzero(idx == t)[0]); wm = w[np.ix_(idx, idx)]
    for i, j in zip(*np.nonzero(np.triu(wm))): ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], '-', color='.75', lw=.3 + 2 * wm[i, j] / w.max(), zorder=1)
    ax.scatter(p[role[idx], 0], p[role[idx], 1], s=14, c='.5', edgecolors=INK, linewidths=.3, zorder=2); ax.scatter(p[~role[idx], 0], p[~role[idx], 1], s=8, c='w', edgecolors=INK, linewidths=.3, zorder=2); ax.scatter([p[kk, 0]], [p[kk, 1]], s=45, c=blue, edgecolors=INK, linewidths=.4, zorder=3); ax.set_aspect('equal', adjustable='datalim'); frame(ax)
    ax = fig.add_axes([xs[3], y1, W, Hh]); rows = idx[np.argsort(hb[m], kind='stable')][:12]; ax.imshow(Z[b, 0, rows, :7], cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto'); frame(ax)
    ax = fig.add_axes([xs[4], y1, W, Hh]); e = np.zeros((len(hb), 1)); e[t] = 1
    for tt in net.scales: h = H(tt, e)[:, 0]; ax.plot(range(6), [h[hb == i].mean() for i in range(6)], marker='o', ms=2, lw=1)
    ax.set_yscale('log'); frame(ax)
    ax = fig.add_axes([xs[5], y1, W, Hh]); V0 = S[b, 0, t, :, :7]; ax.imshow(V0, cmap='Blues', vmin=0, vmax=V0.max(), aspect='auto'); frame(ax)
    for x, (title, call) in zip(xs, (('paired cells, trap atom,\nfield reductions', 'read_structure_files\npotentials, shares'), ('settings', 'Network\nnetwork.json'), ('contact graph', 'contact_graph\ngraph_weights'),
                                    ('channels x,\nstandardised', 'structural_channels\n+ potential, electron'), ('operator,\nwavelets $W_t$', 'operator'), ('scattering\n$S_t = H_t\\,|W_t x|$', 'scattering'))): label(fig, x + W / 2, y1 - .025, title, call)
    for i in range(5): arrow(fig, (xs[i] + W + .004, y1 + Hh / 2), (xs[i + 1] - .004, y1 + Hh / 2))
    # bottom row: the fingerprint at the trap atom and its readings, the descriptors and the model, the outputs
    xb = [.10, .40, .70]; Wb, Hb, y2 = .20, .27, .10
    ax = fig.add_axes([xb[0], y2, Wb, Hb])
    for c in CLASSES: ax.scatter(ev.T0__structure[K[c]], ev.eta__structure[K[c]], s=7, c=COL[c], marker=MARK[c], edgecolors=INK, linewidths=.2)
    frame(ax); ax.text(.97, .04, r'$T_0$', transform=ax.transAxes, ha='right', va='bottom', fontsize=7); ax.text(.97, .96, r'$\eta$', transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax = fig.add_axes([xb[1], y2, Wb, Hb])
    for c in CLASSES: ax.scatter(ev.stabilisation_eV[K[c]], ev.stabilisation_predicted_eV[K[c]], s=7, c=COL[c], marker=MARK[c], edgecolors=INK, linewidths=.2)
    lim = [ev.stabilisation_eV.min() - .1, ev.stabilisation_eV.max() + .1]; ax.plot(lim, lim, ':', color=GREY, lw=.8); frame(ax); ax.text(.97, .04, 'stabilisation', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5); ax.text(.04, .96, 'predicted', transform=ax.transAxes, ha='left', va='top', fontsize=6.5)
    ax = fig.add_axes([xb[2], y2, Wb, Hb]); ax.axis('off'); ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='square,pad=0', fc='w', ec=INK, lw=.8, transform=ax.transAxes))
    ax.text(.08, .92, "events.csv\nfingerprint.npz\nclass_separation.csv\nidentification.csv\nmodel.csv\nfigures", transform=ax.transAxes, va='top', fontsize=6.3, family='monospace')
    for x, (title, call) in zip(xb, (('fingerprint at the trap atom, $V_0$ and $V^-$;\n' + r'readings $T_0$, $\eta$, $\eta_t$, reach', 'demo/fingerprint.py'), ('descriptors and\nheld-out model', 'event_descriptors\nmodels.nested'), ('outputs', 'demo/analyse.py  demo/plot.py'))): label(fig, x + Wb / 2, y2 - .03, title, call)
    for i in range(2): arrow(fig, (xb[i] + Wb + .004, y2 + Hb / 2), (xb[i + 1] - .004, y2 + Hb / 2))
    yg = .44; xe = xs[5] + W + .014; fig.add_artist(Line2D([xs[5] + W + .004, xe, xe, xb[0] + Wb / 2], [y1 + Hh / 2, y1 + Hh / 2, yg, yg], transform=fig.transFigure, color=INK, lw=.8)); arrow(fig, (xb[0] + Wb / 2, yg), (xb[0] + Wb / 2, y2 + Hb + .004))
    save(fig, OUT, FIG, 'fig00_workflow')


if __name__ == '__main__':
    main()
