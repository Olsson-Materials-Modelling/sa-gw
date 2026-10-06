"""Held-out regression on descriptor blocks with every choice made inside the training cells."""
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold


def prepare(Xtr, Xte, gaussianise=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.); tr = np.where(np.isfinite(Xtr), Xtr, med); te = np.where(np.isfinite(Xte), Xte, med)
    if gaussianise:                                               # a held-out value takes the normal score of its place among the training values
        n = len(tr); srt = np.sort(tr, axis=0); score = norm.ppf((np.arange(1, n + 1) - .5) / n); a, b = np.empty_like(tr), np.empty_like(te)
        for j in range(tr.shape[1]):
            xs, ix = np.unique(srt[:, j], return_index=True); cnt = np.diff(np.r_[ix, n]); ys = np.array([score[i:i + k].mean() for i, k in zip(ix, cnt)])
            a[:, j] = np.interp(tr[:, j], xs, ys); b[:, j] = np.interp(te[:, j], xs, ys)
        tr, te = a, b
    mu = tr.mean(0); sd = tr.std(0); ok = np.flatnonzero(sd > 1e-12); z = (tr[:, ok] - mu[ok]) / sd[ok]; keep = []; seen = set()
    for j in range(z.shape[1]):                                   # a column and its negative count as one
        v = z[:, j]; nz = np.flatnonzero(abs(v) > 1e-8); sign = 1. if not len(nz) else np.sign(v[nz[0]]); key = np.round(sign * v, 9) + 0.
        if key.tobytes() not in seen: keep.append(j); seen.add(key.tobytes())
    cols = ok[keep]; return (tr[:, cols] - mu[cols]) / sd[cols], (te[:, cols] - mu[cols]) / sd[cols]


def smoother(Z, V, kind, alpha):
    """Prediction matrix of ridge ('ridge') or Gaussian-kernel ('rbf') regression with an unpenalised mean."""
    p = max(Z.shape[1], 1); K = Z @ Z.T / p; Q = V @ Z.T / p
    if kind == 'rbf':
        zz = (Z * Z).sum(1); K = np.exp(-np.maximum(zz[:, None] + zz[None, :] - 2 * (Z @ Z.T), 0) / p); Q = np.exp(-np.maximum((V * V).sum(1)[:, None] + zz[None, :] - 2 * (V @ Z.T), 0) / p)
    A = cho_factor(K + alpha * np.eye(len(K)), lower=True, check_finite=False); H = cho_solve(A, Q.T, check_finite=False).T
    if kind == 'rbf':                                             # unpenalised mean b = 1'A^-1 y / 1'A^-1 1
        a1 = cho_solve(A, np.ones(len(K)), check_finite=False); return H + (1 - H.sum(1))[:, None] * a1[None, :] / a1.sum()
    return H + (1 - H.sum(1))[:, None] / len(Z)


def nested(blocks, grid, y, strata, fold, seed, gaussianise=False, inner=4):
    """Held-out predictions with block, kernel and penalty chosen inside the training folds."""
    names = list(blocks); n = len(y); nf = int(fold.max()) + 1; pred = np.zeros((len(names) * len(grid), n)); loss = np.zeros((nf, len(names) * len(grid)))
    for outer in range(nf):
        tr, te = np.flatnonzero(fold != outer), np.flatnonzero(fold == outer); splits = list(StratifiedKFold(inner, shuffle=True, random_state=seed + outer).split(tr, strata[tr]))
        for bi, nm in enumerate(names):
            X = blocks[nm]; prep = [(tr[a], tr[b], *prepare(X[tr[a]], X[tr[b]], gaussianise)) for a, b in splits]; z0, v0 = prepare(X[tr], X[te], gaussianise)
            for gi, (kind, al) in enumerate(grid):
                k = bi * len(grid) + gi; pred[k, te] = smoother(z0, v0, kind, al) @ y[tr]
                for a, b, z, v in prep: loss[outer, k] += ((y[b] - smoother(z, v, kind, al) @ y[a]) ** 2).sum()
    oof = np.zeros(n); chosen = []; per_block = {nm: np.zeros(n) for nm in names}
    for outer in range(nf):
        te = fold == outer; k = int(np.argmin(loss[outer])); oof[te] = pred[k, te]; chosen.append(dict(fold=outer, block=names[k // len(grid)], kind=grid[k % len(grid)][0], penalty=float(grid[k % len(grid)][1])))
        for bi, nm in enumerate(names):
            ks = np.arange(bi * len(grid), (bi + 1) * len(grid)); per_block[nm][te] = pred[ks[int(np.argmin(loss[outer, ks]))], te]
    return oof, chosen, per_block
