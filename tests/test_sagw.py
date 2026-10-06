"""Checks of the library on constructed inputs and on the example data.  pytest -q"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import expm
from scipy.integrate import quad
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files, distances, displacement
from sagw.graph import contact_graph, graph_weights, laplacian, operator, scattering, structural_channels, bonds_from
from sagw.network import Network, RATIOS
from sagw.descriptors import event_descriptors
from sagw.potential import site_and_shell, smooth
from sagw.spin import voronoi_integral, shares, carries

DATA = ROOT / 'example/data'; NET = Network.load(DATA / 'network.json'); SCALES = NET.scales


def cell_zero():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); return boxes, sym[0], cell[0], pos[0]


def test_example_data_is_paired():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv')
    assert len(boxes) == 100 and list(ev.cell) == boxes and pos.shape[1:3] == (2, 119) and ev.kind.value_counts().to_dict() == {k: 20 for k in ('dangling bond', 'bond break', 'polaron', 'substitutional', 'interstitial')}
    assert set(np.unique(sym)) <= set(NET.expected) and (ev.precursor_atom != ev.trap_atom).sum() == 4


def test_classes_follow_the_contact_graph():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); t = ev.trap_atom.to_numpy() - 1; k = ev.kind.to_numpy()
    z = np.array([[int(contact_graph(distances(pos[b, q], cell[b]), NET.role_of(sym[b]), NET).sum(1)[t[b]]) for q in range(2)] for b in range(len(boxes))]); species = sym[np.arange(len(boxes)), t]
    assert (z[k == 'dangling bond', 0] == 3).all() and (z[k == 'bond break', 0] == 4).all() and (z[k == 'bond break', 1] <= 3).all() and (z[k == 'polaron'] == 4).all()
    assert (z[k == 'substitutional'] == 4).all() and (species[k == 'substitutional'] == 'X').all() and (z[k == 'interstitial', 0] == 2).all() and (z[k == 'interstitial', 1] == 3).all() and (species[k == 'interstitial'] == 'I').all()
    assert (species[np.isin(k, ['dangling bond', 'bond break', 'polaron'])] == 'A').all()


def test_example_physics():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); k = ev.kind.to_numpy(); b, a = ev.level_before_eV.to_numpy(), ev.level_after_eV.to_numpy()
    assert (b < 0).all() and (a < b).all() and (a > -8).all() and (b[np.isin(k, ['polaron', 'bond break'])] > -1.0).all() and (b[k == 'dangling bond'] > -4).all() and (a[k == 'dangling bond'] < -4).all()
    assert np.quantile(a[k == 'interstitial'], .9) < np.quantile(a[k == 'dangling bond'], .1) and np.quantile(b[k == 'substitutional'], .1) > np.quantile(b[k == 'dangling bond'], .9)
    assert (b - a)[np.isin(k, ['substitutional', 'interstitial'])].max() < 0.55 and abs(np.median(a[k == 'bond break']) - np.median(a[k == 'dangling bond'])) < 0.5
    e = pd.read_parquet(DATA / 'electron.parquet'); neff = e.groupby(['cell', 'state']).share.apply(lambda s: shares(s.to_numpy())[1]).unstack().loc[ev.cell]
    assert (neff.P[np.isin(k, ['polaron', 'bond break'])] > 10).all() and (neff.P[~np.isin(k, ['polaron', 'bond break'])] < 4).all() and (neff['T'][k != 'polaron'] < 3).all() and neff['T'][k == 'polaron'].between(2.5, 8).all()
    rel = ev.precursor_atom != ev.trap_atom; assert (k[rel] == 'dangling bond').all()
    for b_ in np.flatnonzero(rel): assert sym[b_][ev.precursor_atom[b_] - 1] == 'A' and contact_graph(distances(pos[b_, 0], cell[b_]), NET.role_of(sym[b_]), NET).sum(1)[ev.precursor_atom[b_] - 1] == 3


def test_network_settings_follow_the_bond_length():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); m = Network.from_structures(NET.role_a, NET.expected, NET.ideal_cosine, sym[:20], pos[:20, 0], cell[:20])
    assert abs(m.bond - NET.bond) < 0.08 and 1.2 * m.bond < m.cutoff < 1.5 * m.bond and abs(m.sigma / m.bond - RATIOS['sigma']) < 1e-9
    assert NET.cutoff == RATIOS['cutoff'] * NET.bond and NET.spheres == tuple(r * NET.bond for r in RATIOS['spheres']) and NET.shell == tuple(r * NET.bond for r in RATIOS['shell'])
    assert (NET.role_of(np.array(['A', 'B', 'X', 'I'])) == [True, False, True, True]).all()


def test_contacts_join_unlike_roles_only():
    _, sym, cell, pos = cell_zero(); role = NET.role_of(sym); c = contact_graph(distances(pos[0], cell), role, NET)
    assert not (c & (role[:, None] == role[None, :])).any() and set(NET.role_a) == {s for s in NET.expected if NET.role_of(np.array([s]))[0]}


def test_every_atom_is_on_the_graph():
    boxes, sym, cell, pos = read_structure_files(DATA / 'structures')
    assert all((graph_weights(d := distances(pos[b, q], cell[b]), contact_graph(d, NET.role_of(sym[b]), NET), NET).sum(1) > 0).all() for b in range(len(boxes)) for q in range(2))


def test_operator_is_the_matrix_exponential_and_averages():
    _, sym, cell, pos = cell_zero(); d = distances(pos[0], cell); w = graph_weights(d, contact_graph(d, NET.role_of(sym), NET), NET); n = len(w); L, deg = laplacian(w); H = operator(w)
    Hm = np.column_stack([H(2.0, np.eye(n)[:, [k]]) for k in range(n)]); s = 1 / np.sqrt(deg)
    assert np.abs(Hm - s[:, None] * expm(-2 * L) * np.sqrt(deg)[None, :]).max() < 1e-10
    assert np.abs(Hm.sum(1) - 1).max() < 1e-10 and Hm.min() > -1e-12
    one = np.ones((n, 1)); assert max(np.abs(H(t, one) - H(2 * t, one)).max() for t in SCALES) < 1e-10


def test_scattering_is_H_abs_W_and_equivariant():
    _, sym, cell, pos = cell_zero(); d = distances(pos[0], cell); w = graph_weights(d, contact_graph(d, NET.role_of(sym), NET), NET); x = structural_channels(pos[0], cell, sym, NET); x = (x - x.mean(0)) / (x.std(0) + 1e-12); H = operator(w)
    S = scattering(H, x, SCALES); h = {t: H(t, x) for t in (.5, 1, 2, 4, 8, 16)}
    assert max(np.abs(H(t, np.abs(h[t] - h[2 * t])) - S[:, k]).max() for k, t in enumerate(SCALES)) < 1e-10 and S.min() >= 0
    perm = np.random.default_rng(3).permutation(len(w)); Sp = scattering(operator(w[np.ix_(perm, perm)]), x[perm], SCALES); assert np.abs(Sp - S[perm]).max() < 1e-9


def test_structural_channels_count_contacts():
    _, sym, cell, pos = cell_zero(); d = distances(pos[0], cell); role = NET.role_of(sym); F = structural_channels(pos[0], cell, sym, NET); k = np.array([NET.expected[s] for s in sym])
    assert np.array_equal(F[:, 0] - F[:, 1], k - contact_graph(d, role, NET).sum(1)) and (F[:, 2] > 1.3).all() and (F[:, 4] >= F[:, 2]).all()


def test_displacement_wavelets_integrate_to_the_graph_deformation():
    _, sym, cell, pos = cell_zero(); d = distances(pos[0], cell); w = graph_weights(d, contact_graph(d, NET.role_of(sym), NET), NET); L, deg = laplacian(w); u = displacement(pos[0], pos[1], cell)
    x = np.sqrt(deg)[:, None] * u; qd = (x * (L @ x)).sum(); lam, U = np.linalg.eigh(L); coef = ((U.T @ x) ** 2).sum(1)
    integral = sum(c * quad(lambda t, l=l: ((np.exp(-t * l) - np.exp(-2 * t * l)) / t) ** 2, 0, np.inf)[0] for l, c in zip(lam, coef) if l > 1e-9)
    assert abs(integral / ((10 * np.log(2) - 6 * np.log(3)) * qd) - 1) < 1e-6
    assert abs(qd - .5 * (w * ((u[:, None] - u[None]) ** 2).sum(2)).sum()) < 1e-9


def test_descriptors_ignore_a_rigid_translation():
    _, sym, cell, pos = cell_zero(); ev = pd.read_csv(DATA / 'events.csv'); t = int(ev.trap_atom[0]) - 1
    a = event_descriptors(pos[0], pos[1], cell, sym, t, NET); b = event_descriptors(pos[0], pos[1] + np.array([.18, -.23, .11]), cell, sym, t, NET)
    worst = max(abs(a[c] - b[c]) / (1e-12 + abs(a[c])) for c in a if c.startswith(('wavelet__', 'quadratic__')) and np.isfinite(a[c])); assert worst < 1e-8 and len(a) == 215


def test_field_reductions_on_a_uniform_grid():
    grid = np.full((36, 36, 36), 2.5); step = np.full(3, 8.0 / 36); p3 = np.array([[1., 1., 1.], [4., 4., 4.], [6., 2., 5.]]); site, shell = site_and_shell(grid, step, p3, 1.0, (1.4, 3.0))
    assert np.abs(site - 2.5).max() < 1e-10 and np.abs(shell - 2.5).max() < 1e-10 and abs(voronoi_integral(np.ones((36, 36, 36)), step, p3).sum() - 8.0 ** 3) < 1e-8
    n = 36; h = np.full(3, .25 / 0.529177210903); x = np.arange(n) * .25; wave = np.cos(2 * np.pi * 6 * x / (n * .25))[:, None, None] * np.ones((1, n, n)); amp = np.exp(-.5 * (.15 * 2 * np.pi * 6 / 9) ** 2)
    assert abs(np.abs(smooth(wave, h)).max() - amp) < 1e-10 and np.abs(smooth(np.full((n, n, n), 2.5), h) - 2.5).max() < 1e-10


def test_shares_and_relocation():
    s = np.array([0.5, 0.3, 0.1, 0.05, 0.05]); w, n_eff = shares(s); assert abs(w.sum() - 1) < 1e-12 and abs(n_eff - 1 / (w ** 2).sum()) < 1e-12
    assert carries(s, 0)[0] and carries(s, 1)[0] and not carries(s, 4)[0] and carries(s, 4)[1] == 4


def test_bonds_from_counts_shortest_paths():
    _, sym, cell, pos = cell_zero(); c = contact_graph(distances(pos[0], cell), NET.role_of(sym), NET); h = bonds_from(c, 0); assert h[0] == 0 and set(h[c[0]]) == {1} and h.max() <= 6


def test_generator_is_deterministic():
    sys.path.insert(0, str(ROOT / 'example')); import make_example as M
    a = M.network(np.random.default_rng(1), 20, 39, 1); b = M.network(np.random.default_rng(1), 20, 39, 1)
    assert (a[0] == b[0]).all() and np.abs(a[1] - b[1]).max() == 0 and np.abs(a[2] - b[2]).max() == 0
