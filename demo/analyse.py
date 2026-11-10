"""Trap-type identification, pairwise separation, response signatures, relocation check, cutoff sensitivity and the held-out descriptor model of the
example events (demo/out/*.csv)."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files
from sagw.network import Network
from sagw.sensitivity import sensitivity
from sagw import descriptors, models
from sagw.spin import carries, shares
sys.path.insert(0, str(ROOT / 'demo')); from style import classes_of

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'; SEED = 0; REPEATS = 10; BOOT = 1000
PENALTIES = [('ridge', a) for a in np.logspace(-3, 2, 16)]


def held_out(X, y, seed, folds=5):
    """Held-out class probabilities of a multinomial logistic regression, standardised inside the training folds."""
    p = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; m = LogisticRegression(C=1.0, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]); p[te] = m.predict_proba((X[te] - mu) / sd)
    return p


def auc(y, s):
    """Area under the ROC curve of scores s for labels y (1 the positive class), ties counting one half."""
    r = rankdata(s); n1 = int(y.sum()); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def separation(x, y, a, b, rng):
    """Held-out AUC of one summary value or vector for two trap types (a single value takes its sign inside the training folds) and the 95 % interval
    from a bootstrap of the held-out scores within each type."""
    k = np.isin(y, [a, b]); yk = (y[k] == b).astype(int); s = held_out(x[k].reshape(k.sum(), -1), yk, SEED)[:, 1]; one, zero = np.flatnonzero(yk == 1), np.flatnonzero(yk == 0)
    boot = [auc(yk[ix], s[ix]) for ix in (np.r_[rng.choice(one, len(one)), rng.choice(zero, len(zero))] for _ in range(BOOT))]; return auc(yk, s), *np.quantile(boot, [.025, .975])


def main(out=OUT):
    net = Network.load(DATA / 'network.json'); scales = net.scales; ev = pd.read_csv(out / 'events.csv'); z = np.load(out / 'fingerprint.npz'); S = z['S'].astype(float); trap = z['trap']; y = ev.kind.to_numpy(); nb = len(ev); ar = np.arange(nb); classes = classes_of(y)
    _, sym, cell, pos = read_structure_files(DATA / 'structures'); V0, V1 = S[ar, 0, trap], S[ar, 1, trap]; struct = lambda v: v[:, :, :7].reshape(nb, -1); flat = lambda v: v.reshape(nb, -1)
    # trap-type identification, held out, against bond counts and the size of the relaxation
    sets = {'bond counts and deformation': ev[['contacts_before', 'contacts_after', 'graph_deformation_A2', 'max_displacement_A']].to_numpy(float),
            'neutral fingerprint V0 (structure)': struct(V0), 'response V- - V0 (structure)': struct(V1 - V0), 'neutral and response (structure)': np.hstack([struct(V0), struct(V1 - V0)]),
            'full fingerprint (all families)': np.hstack([flat(V0), flat(V1 - V0)])}
    rows, conf = [], []
    for name, X in sets.items():
        acc = []
        for r in range(REPEATS):
            p = held_out(X, y, SEED + r); pred = np.array(sorted(classes))[p.argmax(1)]; acc.append((pred == y).mean())
            if r == 0: conf += [dict(features=name, true=a, predicted=b, count=c) for a, row in zip(classes, confusion_matrix(y, pred, labels=classes)) for b, c in zip(classes, row)]
        rows.append(dict(features=name, n_features=X.shape[1], accuracy=np.mean(acc), accuracy_sd=np.std(acc), chance=1 / len(classes)))
    pd.DataFrame(rows).to_csv(out / 'identification.csv', index=False); pd.DataFrame(conf).to_csv(out / 'confusion.csv', index=False)
    # pairwise separation of every summary value, held out, with bootstrap intervals
    readings = {'T0': ev['T0__structure'].to_numpy(), 'eta': ev['eta__structure'].to_numpy(), 'neutral fingerprint V0': struct(V0), 'response V- - V0': struct(V1 - V0), 'full fingerprint (all families)': sets['full fingerprint (all families)']}
    rng = np.random.default_rng(SEED); rows = []
    for name, x in readings.items():
        for a, b in combinations(classes, 2):
            v, lo, hi = separation(x, y, a, b, rng); rows.append(dict(reading=name, classes=f'{a} | {b}', auc=v, ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(out / 'class_separation.csv', index=False)
    # response signature of each trap type: median standardised response at the trap atom, each event scaled to unit size
    R = flat(V1 - V0); R = R / np.linalg.norm(R, axis=1, keepdims=True); R = (R - R.mean(0)) / (R.std(0) + 1e-12); nc = len(z['channels'])
    pd.DataFrame([dict(kind=c, scale=t, channel=ch, response=float(np.median(R[y == c, k * nc + j]))) for c in classes for k, t in enumerate(scales) for j, ch in enumerate(z['channels'])]).to_csv(out / 'response_signature.csv', index=False)
    # relocation: is the electron already on the trap atom before relaxation?
    e = pd.read_parquet(DATA / 'electron.parquet'); neff = e.groupby(['cell', 'state']).share.apply(lambda s: shares(s.to_numpy())[1]).unstack().loc[ev.cell]; ev['n_eff_before'] = neff.P.to_numpy(); ev['n_eff_after'] = neff['T'].to_numpy()
    a = np.load(DATA / 'potentials.npz'); ev['site_potential_before_eV'] = a['site_eV'][ar, 0, trap]; ev['site_potential_after_eV'] = a['site_eV'][ar, 1, trap]
    e = e[e.state == 'P']
    flag = np.array([not carries(g.sort_values('atom').share.to_numpy(), int(t) - 1)[0] for (_, g), t in zip(e.groupby('cell', sort=True), ev.set_index('cell').trap_atom.loc[sorted(e.cell.unique())])])
    planted = (ev.precursor_atom != ev.trap_atom).to_numpy(); ev['relocation_flagged'] = flag
    pd.DataFrame([dict(planted=p, flagged=f, events=int(((planted == p) & (flag == f)).sum())) for p in (True, False) for f in (True, False)]).to_csv(out / 'relocation.csv', index=False)
    # reach of each scale in A; sensitivity to the cutoff and the weight width (5 %)
    pd.DataFrame([dict(scale=t, **{k: float(np.quantile(ev[f'scale_radius_t{t:g}_A'], q)) for k, q in (('radius_median_A', .5), ('radius_q1_A', .25), ('radius_q3_A', .75))}) for t in scales]).to_csv(out / 'scale_radius.csv', index=False)
    summary, flips = sensitivity(sym, pos, cell, trap, net)
    for c in classes: summary[f'contact_flips__{c}'] = [int(flips[k][y == c].sum()) for k in flips.columns]
    summary.to_csv(out / 'sensitivity.csv', index=False)
    # descriptor model of the stabilisation energy
    X = pd.DataFrame([descriptors.event_descriptors(pos[b, 0], pos[b, 1], cell[b], sym[b], trap[b], net) for b in range(nb)]); X.to_csv(out / 'event_descriptors.csv', index=False)
    yv = ev.stabilisation_eV.to_numpy(); fold = np.zeros(nb, int)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)): fold[te] = f
    oof, _, _ = models.nested({'structural response descriptors': X.to_numpy(float)}, PENALTIES, yv, y, fold, SEED)
    qd = ev.graph_deformation_A2.to_numpy()[:, None]; oracle = np.hstack([qd, pd.get_dummies(y, drop_first=True).to_numpy(float)]); base, orc = np.zeros(nb), np.zeros(nb)
    for f in range(5):
        base[fold == f] = LinearRegression().fit(qd[fold != f], yv[fold != f]).predict(qd[fold == f]); orc[fold == f] = LinearRegression().fit(oracle[fold != f], yv[fold != f]).predict(oracle[fold == f])
    ev['stabilisation_predicted_eV'] = oof; ev.to_csv(out / 'events.csv', index=False)
    pd.DataFrame([dict(model='structural response descriptors (ridge, held out)', inputs=X.shape[1], r2=r2_score(yv, oof)), dict(model='graph deformation alone', inputs=1, r2=r2_score(yv, base)),
                  dict(model='graph deformation and the trap type', inputs=len(classes), r2=r2_score(yv, orc))]).to_csv(out / 'model.csv', index=False)


if __name__ == '__main__':
    main()
