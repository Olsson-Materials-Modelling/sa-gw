#!/usr/bin/env python3
"""Synthetic example data: random network glasses (four-fold A bridged by two-fold B), one trapping event per cell of five kinds, each with a level, a well and an electron share.
    python3 example/make_example.py [out_dir]     (deterministic; fifteen minutes)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sagw.structures import minimum_image, separations, distances
from sagw.graph import contact_graph, graph_weights, bonds_from, structural_channels, operator, scattering
from sagw.network import Network

SEED = 7; N_A, N_B = 40, 79; NUMBER_DENSITY = 0.066; R0 = 1.62                 # atoms per A3 and bond length (A) of a typical network glass
NET = Network.from_bond_length(role_a=('A', 'X', 'I'), expected={'A': 4, 'B': 2, 'X': 4, 'I': 2}, ideal_cosine={'A': -1 / 3, 'B': -0.82, 'X': -1 / 3, 'I': -1.0}, bond=R0)   # the settings every script reads (data/network.json)
CLASSES = ('dangling bond', 'bond break', 'polaron', 'substitutional', 'interstitial'); PER_CLASS = 20; MADE_PER_CLASS = 30; RELOCATE = 4     # of the cells made, the ones nearest the class median of the readings are kept
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'data'
GAP = 8.0                                                                   # eV; levels are eV below the conduction band edge
LEVEL = {'dangling bond': ((-2.5, 0.40), (-5.6, 0.30)), 'bond break': ((-0.3, 0.15), (-5.6, 0.30)), 'polaron': ((-0.15, 0.10), (-4.0, 0.40)),
         'substitutional': ((-1.3, 0.30), (-0.3, 0.07)), 'interstitial': ((-6.8, 0.20), (-0.3, 0.07))}         # the empty level before trapping (mean, spread); after: the occupied level, or its shift for a fixed impurity level
Q_REF = {'dangling bond': 0.12, 'bond break': 0.5, 'polaron': 0.3}       # nominal graph deformation (A2); an intrinsic level deepens with the log of the deformation
WELL = {'dangling bond': ((-1.2, 1.6), (-1.3, 1.6)), 'bond break': ((-0.35, 0.5), (-1.15, 1.5)), 'polaron': ((-0.3, 0.5), (-1.0, 0.7)),
        'substitutional': ((-0.9, 1.0), (-0.95, 1.0)), 'interstitial': ((-1.3, 1.2), (-1.35, 1.2))}           # potential at the site and confinement (eV) at the trap atom, before and after
WIDTH = {'dangling bond': (0.4, 0.4), 'bond break': (3.0, 0.45), 'polaron': (3.0, None), 'substitutional': (0.4, 0.4), 'interstitial': (0.35, 0.35)}   # decay of the electron share (bonds), before and after


def bonds_of(pos, cell, role, want):
    """Bond list (closest A-B pairs first, while both atoms have free valence) and the angle triplets at every atom."""
    n = len(pos); d = distances(pos, cell); ii, jj = np.nonzero(np.triu(role[:, None] != role[None, :])); order = np.argsort(d[ii, jj]); free = want.copy(); b = []
    for k in order:
        i, j = ii[k], jj[k]
        if free[i] > 0 and free[j] > 0 and d[i, j] < 3.6: b.append((i, j)); free[i] -= 1; free[j] -= 1
    for i in set(range(n)) - set(np.array(b).ravel()):                     # an atom left without a bond takes its nearest unlike atom
        opp = np.flatnonzero(role != role[i]); j = int(opp[np.argmin(d[i, opp])]); b.append((min(i, j), max(i, j)))
    b = np.array(sorted(b)); tri = []
    for i in range(n):
        nb = sorted(set(b[b[:, 0] == i, 1]) | set(b[b[:, 1] == i, 0]))
        tri += [(i, a, c) for x, a in enumerate(nb) for c in nb[x + 1:]]
    return b, np.array(tri)


def energy(x, cell, role, b, tri, cos0, ka, kb=12.0, kr=4.0):
    """Model energy and gradient: bond springs, non-bonded repulsion, angle springs (stiff at A, soft at the bridging B)."""
    n = len(role); p = x.reshape(n, 3); v = separations(p, cell); d = np.linalg.norm(v, axis=2) + np.eye(n); g = np.zeros((n, 3)); e = 0.0
    r = d[b[:, 0], b[:, 1]]; u = v[b[:, 0], b[:, 1]] / r[:, None]; f = 2 * kb * (r - R0); e += kb * ((r - R0) ** 2).sum()
    np.add.at(g, b[:, 0], -f[:, None] * u); np.add.at(g, b[:, 1], f[:, None] * u)
    rep = np.where(role[:, None] & role[None, :], 2.9, np.where(role[:, None] | role[None, :], 2.4, 2.5)); close = (d < rep) & ~np.eye(n, dtype=bool); close[b[:, 0], b[:, 1]] = close[b[:, 1], b[:, 0]] = False
    ii, jj = np.nonzero(np.triu(close)); r = d[ii, jj]; u = v[ii, jj] / r[:, None]; f = -2 * kr * (rep[ii, jj] - r); e += kr * ((rep[ii, jj] - r) ** 2).sum()
    np.add.at(g, ii, -f[:, None] * u); np.add.at(g, jj, f[:, None] * u)
    i, a, c = tri.T; va, vc = v[i, a], v[i, c]; ra, rc = d[i, a], d[i, c]; cs = (va * vc).sum(1) / (ra * rc); dif = cs - cos0[i]; e += (ka[i] * dif ** 2).sum()
    dca = vc / (ra * rc)[:, None] - (cs / ra ** 2)[:, None] * va; dcc = va / (ra * rc)[:, None] - (cs / rc ** 2)[:, None] * vc; f = (2 * ka[i] * dif)[:, None]
    np.add.at(g, a, f * dca); np.add.at(g, c, f * dcc); np.add.at(g, i, -f * (dca + dcc))
    return e, g.ravel()


def network(rng, n_a, n_b, n_defects, inflate=1.35):
    """One relaxed network, grown in an inflated box and compressed to NUMBER_DENSITY; `n_defects` A atoms stay three-coordinate."""
    n = n_a + n_b; sym = np.array(['A'] * n_a + ['B'] * n_b); role = NET.role_of(sym); L = (n / NUMBER_DENSITY) ** (1 / 3)
    want = np.where(role, 4, 2); want[rng.choice(n_a, n_defects, replace=False)] = 3; cos0 = np.where(role, NET.ideal_cosine['A'], NET.ideal_cosine['B']); ka = np.where(role, 0.8, 0.3)
    cell = np.full(3, L * inflate); pos = np.empty((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(0, cell[0], 3)
            if i == 0 or np.linalg.norm(minimum_image(pos[:i] - p, cell), axis=1).min() > 1.8: pos[i] = p; break

    def relax(pos, cell, rounds):
        for _ in range(rounds):
            b, tri = bonds_of(pos, cell, role, want); res = minimize(energy, pos.ravel(), args=(cell, role, b, tri, cos0, ka), jac=True, method='L-BFGS-B', options=dict(maxiter=200)); pos = res.x.reshape(n, 3) % cell
        return pos

    pos = relax(pos, cell, 10)
    for f in np.linspace(inflate, 1.0, 7)[1:]:
        new = np.full(3, L * f); pos = pos * (new / cell); cell = new; pos = relax(pos, cell, 4)
    return sym, relax(pos, cell, 8), cell


def sites(sym, pos, cell):
    """Three- and four-coordinate A atoms whose next neighbour is clear of the contact cutoff."""
    role = NET.role_of(sym); d = distances(pos, cell); z = contact_graph(d, role, NET).sum(1); o = d + np.where(role[:, None] == role[None, :], 99, 0) + 99 * np.eye(len(sym)); srt = np.sort(o, 1)
    three = np.flatnonzero(role & (z == 3) & (srt[:, 3] > 2.7)); four = np.flatnonzero(role & (z == 4) & (srt[:, 3] < 2.1) & (srt[:, 4] > 2.7)); return z, three, four


def interstitial_site(sym, pos, cell):
    """A position bonded to two B atoms, clear of every other atom, with a third B atom within reach, or None."""
    role = NET.role_of(sym); d = distances(pos, cell); bb = np.flatnonzero(~role); v = separations(pos, cell)
    pairs = sorted([(i, j) for k, i in enumerate(bb) for j in bb[k + 1:] if 2.9 < d[i, j] < 3.2], key=lambda ij: abs(d[ij] - 3.05))
    for i, j in pairs:
        mid = pos[i] + 0.5 * v[i, j]; axis = v[i, j] / d[i, j]; e1 = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
        h = np.sqrt(max(1.9 ** 2 - (d[i, j] / 2) ** 2, 0.01))
        for phi in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            p = (mid + h * (np.cos(phi) * e1 + np.sin(phi) * e2)) % cell; dd = np.linalg.norm(minimum_image(pos - p, cell), axis=1); others = np.delete(dd, [i, j])
            if others.min() > 2.4 and ((dd[bb] > 2.6) & (dd[bb] < 3.0)).any(): return p, i, j                       # a third B atom within reach at a set distance
    return None


def signature(kind, sym, pos, anion, cell, trap):
    """True when the trap atom shows its class in the contact graph and no atom is left without a contact."""
    z = [contact_graph(distances(p, cell), NET.role_of(sym), NET).sum(1) for p in (pos, anion)]; z0, z1 = z[0][trap], z[1][trap]
    ok = {'dangling bond': z0 == 3 and z1 == 3, 'bond break': z0 == 4 and z1 <= 3, 'polaron': z0 == 4 and z1 == 4, 'substitutional': z0 == 4 and z1 == 4, 'interstitial': z0 == 2 and z1 == 3}[kind]
    return bool(ok) and z[0].min() >= 1 and z[1].min() >= 1


def pick(kind, sym, pos, cell, three, four):
    """The trapping site of a class: the most regular three-fold site for a dangling bond, the most strained four-fold site for a bond break, the most regular four-fold site otherwise."""
    F = structural_channels(pos, cell, sym, NET); regular = F[:, 3] + 0.3 * F[:, 6]
    if kind == 'dangling bond': return int(three[np.argmin(regular[three])])
    if kind == 'bond break': return int(four[np.argmax(F[four, 3])])
    return int(four[np.argmin(regular[four])])


def event(rng, sym, pos, cell, kind):
    """Neutral and relaxed-anion structures of one event and its trap atom (the precursor atom is set later)."""
    sym = sym.copy(); pos = pos.copy(); z, three, four = sites(sym, pos, cell); d = distances(pos, cell); v = separations(pos, cell)
    unit = lambda x: x / np.linalg.norm(x); rnd = lambda: unit(rng.normal(size=3))
    if kind == 'interstitial':
        p, i, j = interstitial_site(sym, pos, cell); sym = np.append(sym, 'I'); pos = np.vstack([pos, p]); trap = len(sym) - 1
    else:
        trap = pick(kind, sym, pos, cell, three, four); nbr = np.flatnonzero(contact_graph(d, NET.role_of(sym), NET)[trap])
    if kind == 'substitutional':
        sym[trap] = 'X'; pos[nbr] += 0.18 * v[trap, nbr] / d[trap, nbr][:, None]                                         # a larger impurity: its four bonds are longer
    if kind == 'bond break':
        far = nbr[np.argmax(d[trap, nbr])]; pos[far] += 0.10 * v[trap, far] / d[trap, far]                                # strain: the longest bond is stretched
    role = NET.role_of(sym); d = distances(pos, cell); v = separations(pos, cell); contact = contact_graph(d, role, NET); nbr = np.flatnonzero(contact[trap]); hop = bonds_from(contact, trap); u = rng.normal(0, 0.01, pos.shape)
    out = lambda i: unit(minimum_image(pos[i] - pos[trap], cell))                                                        # outward from the trap atom
    shell = lambda h, a: u.__setitem__(hop == h, u[hop == h] + a * np.array([out(i) for i in np.flatnonzero(hop == h)]).reshape(-1, 3))
    if kind == 'dangling bond':                                            # compact: the trap atom pyramidalises and its three bonds lengthen a little
        u[trap] += 0.15 * unit(-v[trap, nbr].sum(0)); u[nbr] += 0.02 * v[trap, nbr] / d[trap, nbr][:, None]; shell(2, 0.01)
    elif kind == 'bond break':                                             # the strained bond breaks: that B leaves, the A settles on the other three
        far = nbr[np.argmax(d[trap, nbr])]; keep = nbr[nbr != far]; u[far] += (2.75 - d[trap, far]) * unit(v[trap, far]); u[trap] += 0.30 * unit(v[trap, keep].sum(0)); u[keep] += 0.10 * np.array([out(i) for i in keep]); shell(2, 0.04); shell(3, 0.02)
    elif kind == 'polaron':                                                # collective: every bond of the trap atom lengthens and the shells beyond breathe outwards
        u[trap] += 0.05 * rnd(); u[nbr] += 0.15 * v[trap, nbr] / d[trap, nbr][:, None]; [shell(h, a) for h, a in ((2, 0.10), (3, 0.06), (4, 0.04), (5, 0.02))]
    elif kind == 'substitutional':                                         # off-centre: the impurity moves towards two of its neighbours
        pair = nbr[np.argsort(d[trap, nbr])[:2]]; u[trap] += 0.25 * unit(v[trap, pair].sum(0)); u[nbr] += 0.06 * np.array([out(i) for i in nbr]); shell(2, 0.03)
    else:                                                                  # the interstitial binds a third B atom
        cand = np.flatnonzero(~role & ~contact[trap] & (d[trap] > 2.6) & (d[trap] < 3.0)); third = cand[np.argmin(d[trap, cand])]
        u[trap] += (d[trap, third] - 2.0) * unit(v[trap, third]); u[third] += 0.10 * unit(-v[trap, third]); u[nbr] += 0.06 * np.array([out(i) for i in nbr]); shell(2, 0.03)
    return sym, pos, (pos + u) % cell, trap, trap


def potentials_of(rng, sym, pos, anion, cell, trap, kind):
    """Synthetic potential at the site and over the shell (eV) of every atom at both endpoints."""
    n = len(sym); F = [structural_channels(p, cell, sym, NET) for p in (pos, anion)]; hop = [bonds_from(contact_graph(distances(p, cell), NET.role_of(sym), NET), trap) for p in (pos, anion)]
    site = np.stack([-0.8 * F[q][:, 0] + 0.15 * (F[q][:, 2] - R0) + rng.normal(0, 0.10, n) for q in range(2)])          # every atom: deeper where contacts are missing
    contrast = np.stack([0.6 + 0.5 * F[q][:, 0] + rng.normal(0, 0.08, n) for q in range(2)])
    for q in range(2): site[q, trap], contrast[q, trap] = WELL[kind][q][0] + rng.normal(0, 0.08), WELL[kind][q][1] + rng.normal(0, 0.08)
    if kind == 'polaron': site[1, hop[1] == 1] -= 0.45; site[1, hop[1] == 2] -= 0.15                                  # the well that forms spreads over the neighbours
    if kind == 'bond break': site[1, hop[1] == 1] -= 0.2
    return site, site + contrast


def shares_of(rng, sym, pos, anion, cell, trap, precursor, kind):
    """Synthetic share of the added electron on every atom before (on the precursor) and after relaxation (on the trap atom)."""
    role = NET.role_of(sym); n = len(sym); contact = [contact_graph(distances(p, cell), role, NET) for p in (pos, anion)]
    def share(h, width, neighbours=0.35, beyond=1.0):
        w = np.exp(-h / width) * np.where(role, 1.0, 0.35) * rng.lognormal(0, 0.25, n); w[h == 1] *= neighbours / 0.35; w[h >= 2] *= beyond; return w / w.sum()
    P = share(bonds_from(contact[0], precursor), WIDTH[kind][0]); h = bonds_from(contact[1], trap)
    T = share(h, 100.0, 0.6, 0.003) if kind == 'polaron' else share(h, WIDTH[kind][1])                                  # a polaron ends on the atom and its four neighbours
    return P, T


def readings(cells):
    """T0, eta and eta_t of the structural fingerprint at the trap atom of every cell, standardised over the cells made."""
    F = [[structural_channels(p, c['cell'], c['sym'], NET) for p in (c['pos'], c['anion'])] for c in cells]; allF = np.concatenate([f[0] for f in F]); mu, sd = allF.mean(0), allF.std(0); out = []
    for c, f in zip(cells, F):
        S = []
        for q, p in enumerate((c['pos'], c['anion'])):
            d = distances(p, c['cell']); S.append(scattering(operator(graph_weights(d, contact_graph(d, NET.role_of(c['sym']), NET), NET)), (f[q] - mu) / sd, NET.scales)[c['trap']])
        r0, r1 = np.linalg.norm(S[0], axis=1), np.linalg.norm(S[1] - S[0], axis=1); out.append([r0.sum(), r1.sum() / r0.sum()] + list(r1 / r0))
    return np.array(out)


def select(cells):
    """The PER_CLASS cells of each class nearest to the class median of the readings, in the order made."""
    R = readings(cells); kind = np.array([c['kind'] for c in cells]); keep = np.zeros(len(cells), bool)
    for k in CLASSES:
        m = np.flatnonzero(kind == k); z = (R[m] - np.median(R[m], 0)) / (np.median(np.abs(R[m] - np.median(R[m], 0)), 0) + 1e-9); keep[m[np.argsort(np.abs(z).sum(1))[:PER_CLASS]]] = True
    return [c for c, k in zip(cells, keep) if k]


def main():
    rng = np.random.default_rng(SEED); (OUT / 'structures/q0').mkdir(parents=True, exist_ok=True); (OUT / 'structures/q-1').mkdir(parents=True, exist_ok=True); NET.save(OUT / 'network.json')
    kinds = [k for k in CLASSES for _ in range(MADE_PER_CLASS)]; rng.shuffle(kinds); cells = []
    for k, kind in enumerate(kinds):
        while True:
            sym, pos, cell = network(rng, N_A, N_B - (kind == 'interstitial'), 2); z, three, four = sites(sym, pos, cell)
            if not (z.min() >= 1 and z[NET.role_of(sym)].max() <= 4 and len(three) >= 1 and len(four) >= 2 and (kind != 'interstitial' or interstitial_site(sym, pos, cell) is not None)): continue
            sym, pos, anion, trap, precursor = event(rng, sym, pos, cell, kind)
            if signature(kind, sym, pos, anion, cell, trap): break
        site, shell = potentials_of(rng, sym, pos, anion, cell, trap, kind)
        d0 = distances(pos, cell); w = graph_weights(d0, contact_graph(d0, NET.role_of(sym), NET), NET); u = minimum_image(anion - pos, cell); u -= u.mean(0)
        qd = 0.5 * (w * ((u[:, None] - u[None]) ** 2).sum(2)).sum()
        (b_mu, b_sd), (a_mu, a_sd) = LEVEL[kind]; before = float(np.clip(b_mu + rng.normal(0, b_sd), -GAP + 0.3, -0.05))                # every level lies in the gap
        after = a_mu + rng.normal(0, a_sd) - 0.3 * np.log(qd / Q_REF[kind]) if kind in Q_REF else before + a_mu + rng.normal(0, a_sd); after = float(np.clip(after, -GAP + 0.2, before - 0.05))
        e_stab = 0.15 * (before - after) - 0.05 * qd + 0.05 + rng.normal(0, 0.02)                                     # the level drop, less the elastic cost
        cells.append(dict(kind=kind, sym=sym, pos=pos, anion=anion, cell=cell, trap=trap, precursor=precursor, site=site, shell=shell, before=before, after=after, e_stab=e_stab))
        print(f'made {k + 1}/{len(kinds)}', kind, 'trap', trap + 1, sym[trap], 'precursor', precursor + 1, f'level {before:.2f} -> {after:.2f}', f'E_stab {e_stab:.2f}', flush=True)
    kept = select(cells); planted = 0
    for c in kept:                                                              # the electron first sits on the other dangling bond in RELOCATE of the kept cells
        other = np.flatnonzero(NET.role_of(c['sym']) & (contact_graph(distances(c['pos'], c['cell']), NET.role_of(c['sym']), NET).sum(1) == 3) & (np.arange(len(c['sym'])) != c['trap']))
        if c['kind'] == 'dangling bond' and planted < RELOCATE and len(other): c['precursor'] = int(rng.choice(other)); planted += 1
        c['P'], c['T'] = shares_of(rng, c['sym'], c['pos'], c['anion'], c['cell'], c['trap'], c['precursor'], c['kind'])
    rows, pots, els = [], [], []
    for k, c in enumerate(kept):
        name = f'cell_{k + 1:03d}'; sym = c['sym']
        for q, p in (('q0', c['pos']), ('q-1', c['anion'])):
            lat = ' '.join(f'{x:.6f}' for x in np.diag(c['cell']).ravel())
            (OUT / 'structures' / q / f'{name}.extxyz').write_text(f'{len(sym)}\nLattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T" cell={name} charge={0 if q == "q0" else -1}\n' + ''.join(f'{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n' for s, (x, y, z) in zip(sym, p)))
        rows.append(dict(cell=name, kind=c['kind'], trap_atom=c['trap'] + 1, precursor_atom=c['precursor'] + 1, level_before_eV=round(float(c['before']), 4), level_after_eV=round(float(c['after']), 4), stabilisation_eV=round(float(c['e_stab']), 4))); pots.append((c['site'], c['shell']))
        for st in ('P', 'T'): els += [dict(cell=name, state=st, atom=i + 1, species=sym[i], share=float(c[st][i])) for i in range(len(sym))]
    pd.DataFrame(rows).to_csv(OUT / 'events.csv', index=False)
    np.savez_compressed(OUT / 'potentials.npz', cells=np.array([r['cell'] for r in rows]), site_eV=np.array([p[0] for p in pots]), shell_eV=np.array([p[1] for p in pots]))
    pd.DataFrame(els).to_parquet(OUT / 'electron.parquet', index=False); print(f'{len(rows)} cells kept of {len(cells)} made, written to {OUT}')


if __name__ == '__main__':
    main()
