#!/usr/bin/env python3
"""Class identification, pairwise separation, response signatures, relocation and the held-out descriptor model of the example events (demo/out/*.csv)."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files
from sagw.network import Network
from sagw import descriptors, models
from sagw.spin import carries, shares

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'; CLASSES = ['dangling bond', 'bond break', 'polaron', 'substitutional', 'interstitial']; SEED = 0; REPEATS = 10
PENALTIES = [('ridge', a) for a in np.logspace(-3, 2, 16)]


def held_out(X, y, seed, folds=5):
    """Held-out class probabilities of a multinomial logistic regression, standardised inside the training folds."""
    p = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; m = LogisticRegression(C=1.0, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]); p[te] = m.predict_proba((X[te] - mu) / sd)
    return p


def auc_pair(x, y, a, b):
    """Area under the ROC curve of one reading for two classes; a vector is scored held out by logistic regression."""
    k = np.isin(y, [a, b]); xk, yk = x[k], (y[k] == b).astype(int)
    if xk.ndim == 1: v = roc_auc_score(yk, xk); return max(v, 1 - v)
    return roc_auc_score(yk, held_out(xk, yk, SEED)[:, 1])


def main():
    net = Network.load(DATA / 'network.json'); SCALES = net.scales; ev = pd.read_csv(OUT / 'events.csv'); z = np.load(OUT / 'fingerprint.npz', allow_pickle=True); S = z['S'].astype(float); trap = z['trap']; y = ev.kind.to_numpy(); nb = len(ev); ar = np.arange(nb); n = S.shape[2]
    V0, V1 = S[ar, 0, trap], S[ar, 1, trap]; struct = lambda v: v[:, :, :7].reshape(nb, -1); flat = lambda v: v.reshape(nb, -1)
    # identification of the five classes, held out, against the count of contacts and the size of the relaxation
    sets = {'contacts and deformation': ev[['contacts_before', 'contacts_after', 'graph_deformation_A2', 'max_displacement_A']].to_numpy(float),
            'neutral fingerprint V0 (structure)': struct(V0), 'response V- - V0 (structure)': struct(V1 - V0), 'neutral and response (structure)': np.hstack([struct(V0), struct(V1 - V0)]),
            'full fingerprint (all families)': np.hstack([flat(V0), flat(V1 - V0)])}
    rows, conf = [], []
    for name, X in sets.items():
        acc = []
        for r in range(REPEATS):
            p = held_out(X, y, SEED + r); pred = np.array(sorted(CLASSES))[p.argmax(1)]; acc.append((pred == y).mean())
            if r == 0: conf += [dict(features=name, true=a, predicted=b, count=c) for a, row in zip(CLASSES, confusion_matrix(y, pred, labels=CLASSES)) for b, c in zip(CLASSES, row)]
        rows.append(dict(features=name, n_features=X.shape[1], accuracy=np.mean(acc), accuracy_sd=np.std(acc), chance=1 / len(CLASSES)))
    ident = pd.DataFrame(rows); ident.to_csv(OUT / 'identification.csv', index=False); pd.DataFrame(conf).to_csv(OUT / 'confusion.csv', index=False)
    # pairwise separation of every reading, with bootstrap intervals
    readings = {'T0': ev['T0__structure'].to_numpy(), 'eta': ev['eta__structure'].to_numpy(), 'neutral fingerprint V0': struct(V0), 'response V- - V0': struct(V1 - V0), 'full fingerprint (all families)': sets['full fingerprint (all families)']}
    rng = np.random.default_rng(SEED); rows = []
    for name, x in readings.items():
        for a, b in combinations(CLASSES, 2):
            v = auc_pair(x, y, a, b); boot = []
            for _ in range(200):
                ix = rng.integers(0, nb, nb)
                try: boot.append(auc_pair(x[ix], y[ix], a, b))
                except ValueError: pass
            rows.append(dict(reading=name, classes=f'{a} | {b}', auc=v, ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975)))
    sep = pd.DataFrame(rows); sep.to_csv(OUT / 'class_separation.csv', index=False)
    # response signature of each class: median standardised response at the trap atom, each event scaled to unit size
    R = flat(V1 - V0); R = R / np.linalg.norm(R, axis=1, keepdims=True); R = (R - R.mean(0)) / (R.std(0) + 1e-12); nc = len(z['channels']); boxes, sym, cell, pos = read_structure_files(DATA / 'structures')
    pd.DataFrame([dict(kind=c, scale=t, channel=ch, response=float(np.median(R[y == c, k * nc + j]))) for c in CLASSES for k, t in enumerate(SCALES) for j, ch in enumerate(z['channels'])]).to_csv(OUT / 'response_signature.csv', index=False)
    # relocation: the electron before relaxation against the atom that traps
    e = pd.read_parquet(DATA / 'electron.parquet'); neff = e.groupby(['cell', 'state']).share.apply(lambda s: shares(s.to_numpy())[1]).unstack().loc[ev.cell]; ev['n_eff_before'] = neff.P.to_numpy(); ev['n_eff_after'] = neff['T'].to_numpy()
    a = np.load(DATA / 'potentials.npz'); ev['site_potential_before_eV'] = a['site_eV'][ar, 0, trap]; ev['site_potential_after_eV'] = a['site_eV'][ar, 1, trap]; ev['well_contrast_before_eV'] = (a['shell_eV'] - a['site_eV'])[ar, 0, trap]; ev['well_contrast_after_eV'] = (a['shell_eV'] - a['site_eV'])[ar, 1, trap]
    e = e[e.state == 'P']
    flag = np.array([not carries(g.sort_values('atom').share.to_numpy(), int(t) - 1)[0] for (_, g), t in zip(e.groupby('cell', sort=True), ev.set_index('cell').trap_atom.loc[sorted(e.cell.unique())])])
    planted = (ev.precursor_atom != ev.trap_atom).to_numpy(); ev['relocation_flagged'] = flag
    pd.DataFrame([dict(planted=p, flagged=f, events=int(((planted == p) & (flag == f)).sum())) for p in (True, False) for f in (True, False)]).to_csv(OUT / 'relocation.csv', index=False)
    # the reach of each scale in A, and the descriptor model of the stabilisation
    pd.DataFrame([dict(scale=t, **{k: float(np.quantile(ev[f'scale_radius_t{t:g}_A'], q)) for k, q in (('radius_median_A', .5), ('radius_q1_A', .25), ('radius_q3_A', .75))}) for t in SCALES]).to_csv(OUT / 'scale_radius.csv', index=False)
    X = pd.DataFrame([descriptors.event_descriptors(pos[b, 0], pos[b, 1], cell[b], sym[b], trap[b], net) for b in range(nb)]); X.to_csv(OUT / 'event_descriptors.csv', index=False)
    yv = ev.stabilisation_eV.to_numpy(); fold = np.zeros(nb, int)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)): fold[te] = f
    oof, chosen, _ = models.nested({'structural response descriptors': X.to_numpy(float)}, PENALTIES, yv, y, fold, SEED)
    qd = ev.graph_deformation_A2.to_numpy()[:, None]; oracle = np.hstack([qd, pd.get_dummies(y).to_numpy(float)]); base, orc = np.zeros(nb), np.zeros(nb)
    for f in range(5):
        base[fold == f] = LinearRegression().fit(qd[fold != f], yv[fold != f]).predict(qd[fold == f]); orc[fold == f] = LinearRegression().fit(oracle[fold != f], yv[fold != f]).predict(oracle[fold == f])
    ev['stabilisation_predicted_eV'] = oof; ev['stabilisation_from_deformation_eV'] = base; ev.to_csv(OUT / 'events.csv', index=False)
    pd.DataFrame([dict(model='structural response descriptors (ridge, held out)', inputs=X.shape[1], r2=r2_score(yv, oof)), dict(model='graph deformation alone', inputs=1, r2=r2_score(yv, base)),
                  dict(model='graph deformation and the true class', inputs=6, r2=r2_score(yv, orc))]).to_csv(OUT / 'model.csv', index=False)
    pd.set_option('display.width', 250); print(ident.round(3).to_string(index=False)); print(sep.pivot(index='reading', columns='classes', values='auc').round(2).to_string())
    print(pd.read_csv(OUT / 'relocation.csv').to_string(index=False)); print(pd.read_csv(OUT / 'model.csv').round(3).to_string(index=False))
    print(ev.groupby('kind')[['level_before_eV', 'level_after_eV', 'n_eff_before', 'n_eff_after', 'site_potential_before_eV', 'site_potential_after_eV', 'stabilisation_eV']].median().round(2).to_string())


if __name__ == '__main__':
    main()
