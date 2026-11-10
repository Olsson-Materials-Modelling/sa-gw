"""Checks of the library on constructed inputs and on the example data.  pytest -q"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
from scipy.stats import norm
from scipy.linalg import expm
from scipy.integrate import quad
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files, distances, displacement
from sagw.graph import contact_graph, graph_weights, laplacian, operator, scattering, structural_channels, bonds_from
from sagw.network import Network, RATIOS
from sagw.descriptors import event_descriptors
from sagw.potential import site_and_shell, gaussian_smooth, read_cube, potentials, reference_to_species, well, HARTREE
from sagw.spin import voronoi_integral, shares, carries
from sagw.models import prepare, smoother, nested

DATA = ROOT / 'example/data'; NET = Network.load(DATA / 'network.json'); SCALES = NET.scales


def cell_zero():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); return names, sym[0], cell[0], pos[0]


def test_example_data_is_paired():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv')
    assert len(names) == 100 and list(ev.cell) == names and pos.shape[1:3] == (2, 119) and ev.kind.value_counts().to_dict() == {k: 20 for k in ('dangling bond', 'bond break', 'polaron', 'substitutional', 'interstitial')}
    assert set(np.unique(sym)) <= set(NET.expected) and (ev.precursor_atom != ev.trap_atom).sum() == 4


def test_classes_follow_the_contact_graph():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); t = ev.trap_atom.to_numpy() - 1; k = ev.kind.to_numpy()
    z = np.array([[int(contact_graph(distances(pos[b, q], cell[b]), NET.role_of(sym[b]), NET).sum(1)[t[b]]) for q in range(2)] for b in range(len(names))]); species = sym[np.arange(len(names)), t]
    assert (z[k == 'dangling bond', 0] == 3).all() and (z[k == 'bond break', 0] == 4).all() and (z[k == 'bond break', 1] <= 3).all() and (z[k == 'polaron'] == 4).all()
    assert (z[k == 'substitutional'] == 4).all() and (species[k == 'substitutional'] == 'X').all() and (z[k == 'interstitial', 0] == 2).all() and (z[k == 'interstitial', 1] == 3).all() and (species[k == 'interstitial'] == 'I').all()
    assert (species[np.isin(k, ['dangling bond', 'bond break', 'polaron'])] == 'A').all()


def test_example_physics():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); k = ev.kind.to_numpy(); b, a = ev.level_before_eV.to_numpy(), ev.level_after_eV.to_numpy()
    assert (b < 0).all() and (a < b).all() and (a > -8).all() and (b[np.isin(k, ['polaron', 'bond break'])] > -1.0).all() and (b[k == 'dangling bond'] > -4).all() and (a[k == 'dangling bond'] < -4).all()
    assert np.quantile(a[k == 'interstitial'], .9) < np.quantile(a[k == 'dangling bond'], .1) and np.quantile(b[k == 'substitutional'], .1) > np.quantile(b[k == 'dangling bond'], .9)
    assert (b - a)[np.isin(k, ['substitutional', 'interstitial'])].max() < 0.55 and abs(np.median(a[k == 'bond break']) - np.median(a[k == 'dangling bond'])) < 0.5
    e = pd.read_parquet(DATA / 'electron.parquet'); neff = e.groupby(['cell', 'state']).share.apply(lambda s: shares(s.to_numpy())[1]).unstack().loc[ev.cell]
    assert (neff.P[np.isin(k, ['polaron', 'bond break'])] > 10).all() and (neff.P[~np.isin(k, ['polaron', 'bond break'])] < 4).all() and (neff['T'][k != 'polaron'] < 3).all() and neff['T'][k == 'polaron'].between(2.5, 8).all()
    rel = ev.precursor_atom != ev.trap_atom; assert (k[rel] == 'dangling bond').all()
    for b_ in np.flatnonzero(rel): assert sym[b_][ev.precursor_atom[b_] - 1] == 'A' and contact_graph(distances(pos[b_, 0], cell[b_]), NET.role_of(sym[b_]), NET).sum(1)[ev.precursor_atom[b_] - 1] == 3


def test_network_settings_follow_the_bond_length():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); m = Network.from_structures(NET.role_a, NET.expected, NET.ideal_cosine, sym[:20], pos[:20, 0], cell[:20])
    assert abs(m.bond - NET.bond) < 0.08 and 1.2 * m.bond < m.cutoff < 1.5 * m.bond and abs(m.sigma - RATIOS['sigma'] * m.bond) < 1e-6
    assert abs(NET.cutoff - RATIOS['cutoff'] * NET.bond) < 1e-6 and np.allclose(NET.spheres, [r * NET.bond for r in RATIOS['spheres']], rtol=0, atol=1e-6) and np.allclose(NET.shell, [r * NET.bond for r in RATIOS['shell']], rtol=0, atol=1e-6)
    assert Network.load(DATA / 'network.json') == NET and NET.reference == {'X': 'A', 'I': 'A'} and NET.smooth == 0
    assert (NET.role_of(np.array(['A', 'B', 'X', 'I'])) == [True, False, True, True]).all()


def test_contacts_join_unlike_roles_only():
    _, sym, cell, pos = cell_zero(); role = NET.role_of(sym); c = contact_graph(distances(pos[0], cell), role, NET)
    assert not (c & (role[:, None] == role[None, :])).any() and set(NET.role_a) == {s for s in NET.expected if NET.role_of(np.array([s]))[0]}


def test_every_atom_is_on_the_graph():
    names, sym, cell, pos = read_structure_files(DATA / 'structures')
    assert all((graph_weights(d := distances(pos[b, q], cell[b]), contact_graph(d, NET.role_of(sym[b]), NET), NET).sum(1) > 0).all() for b in range(len(names)) for q in range(2))


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
    grid = np.full((36, 36, 36), 2.5); step = np.full(3, 8.0 / 36); p3 = np.array([[1., 1., 1.], [4., 4., 4.], [6., 2., 5.]]); site, shell = site_and_shell(grid, step, p3, 1.0, (1.4, 2.1))
    assert np.abs(site - 2.5).max() < 1e-10 and np.abs(shell - 2.5).max() < 1e-10 and abs(voronoi_integral(np.ones((36, 36, 36)), step, p3).sum() - 8.0 ** 3) < 1e-8
    n = 36; h = np.full(3, .25 / 0.529177210903); x = np.arange(n) * .25; wave = np.cos(2 * np.pi * 6 * x / (n * .25))[:, None, None] * np.ones((1, n, n)); amp = np.exp(-.5 * (.15 * 2 * np.pi * 6 / 9) ** 2)
    assert abs(np.abs(gaussian_smooth(wave, h)).max() - amp) < 1e-10 and np.abs(gaussian_smooth(np.full((n, n, n), 2.5), h) - 2.5).max() < 1e-10


def test_shares_and_relocation():
    s = np.array([0.5, 0.3, 0.1, 0.05, 0.05]); w, n_eff = shares(s); assert abs(w.sum() - 1) < 1e-12 and abs(n_eff - 1 / (w ** 2).sum()) < 1e-12
    assert carries(s, 0)[0] and carries(s, 1)[0] and not carries(s, 4)[0] and carries(s, 4)[1] == 4


def test_bonds_from_counts_shortest_paths():
    _, sym, cell, pos = cell_zero(); c = contact_graph(distances(pos[0], cell), NET.role_of(sym), NET); h = bonds_from(c, 0); assert h[0] == 0 and set(h[c[0]]) == {1} and h.max() <= 6


def test_generator_is_deterministic():
    sys.path.insert(0, str(ROOT / 'example')); import make_example as M
    a = M.network(np.random.default_rng(1), 20, 39, 1); b = M.network(np.random.default_rng(1), 20, 39, 1)
    assert (a[0] == b[0]).all() and np.abs(a[1] - b[1]).max() == 0 and np.abs(a[2] - b[2]).max() == 0


def test_ridge_is_ridge_and_held_out_predictions_never_see_their_fold():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(4); X = rng.normal(size=(80, 6)); y = X @ rng.normal(size=6) + rng.normal(0, .3, 80); Z, V = prepare(X[:60], X[60:])
    ref = Ridge(alpha=2.0 * Z.shape[1]).fit(Z, y[:60]).predict(V); assert np.abs(smoother(Z, V, 'ridge', 2.0) @ y[:60] - ref).max() < 1e-10
    strata = np.repeat([0, 1], 40); fold = np.arange(80) % 4; grid = [('ridge', 0.1), ('ridge', 10.0), ('rbf', 1.0)]; oof = nested({'x': X}, grid, y, strata, fold, 0)[0]
    z = y.copy(); z[fold == 2] += 100; assert np.abs(nested({'x': X}, grid, z, strata, fold, 0)[0][fold == 2] - oof[fold == 2]).max() < 1e-12
    oof2, _, per = nested({'x': X, 'two': X[:, :2]}, grid, y, strata, fold, 0); assert np.abs(per['x'] - nested({'x': X}, grid, y, strata, fold, 0)[0]).max() < 1e-12


def test_settings_survive_a_round_trip(tmp_path):
    net = Network(('A',), {'A': np.int64(4), 'B': 2}, {'A': -1 / 3, 'B': -0.8}, np.float64(1.62), 2.187, 1.1664, smooth=0.05, reference={'X': 'A'}); net.save(tmp_path / 'n.json')
    assert Network.load(tmp_path / 'n.json') == net


def test_site_and_shell_match_every_grid_point_by_periodic_distance():
    n, a = 32, 8.0; h = a / n; g = np.stack(np.meshgrid(*[np.arange(n) * h] * 3, indexing='ij'), -1).reshape(-1, 3); field = np.random.default_rng(8).normal(size=(n, n, n))
    p = np.array([[0.05, 0.05, 7.95], [1.41, 0.31, 0.12], [4.0, 4.1, 3.9]]); dist = np.stack([np.linalg.norm(np.mod(g - x + a / 2, a) - a / 2, axis=1) for x in p]); clear = dist.min(0) > 0.6
    site, shell = site_and_shell(field, np.full(3, h / 0.529177210903), p / 0.529177210903, 0.8, (1.0, 2.0), core=0.6, width=0.0); f = field.ravel()
    assert np.allclose(site, [f[d < 0.8].mean() for d in dist], rtol=0, atol=1e-12) and np.allclose(shell, [f[(d >= 1.0) & (d < 2.0) & clear].mean() for d in dist], rtol=0, atol=1e-12)


def test_voronoi_cells_and_the_reduction_of_a_cube():
    n = 20; step = np.full(3, 0.5); p = np.array([[0.2, 0.2, 0.2], [9.9, 5.0, 5.0]]); g = np.stack(np.meshgrid(*[np.arange(n) * 0.5] * 3, indexing='ij'), -1)
    near = np.linalg.norm(np.mod(g - p[0] + 5, 10) - 5, axis=-1) < np.linalg.norm(np.mod(g - p[1] + 5, 10) - 5, axis=-1); v = voronoi_integral(near.astype(float), step, p)
    assert abs(v[0] - near.sum() * 0.125) < 1e-12 and v[1] == 0
    m = 24; x = np.arange(m) * 12.0 / m; field = (0.01 * np.cos(2 * np.pi * x / 12.0))[:, None, None] * np.ones((1, m, m)); b = 12.0 / m / 0.529177210903
    atoms = np.array([[1.0, 2.0, 3.0], [7.0, 4.0, 5.0], [3.0, 9.0, 1.0], [10.0, 6.0, 8.0]]); sym = np.array(['A', 'B', 'A', 'B']); z = [14, 7, 14, 7]
    head = f'c\nv\n{4:5d}' + f'{0:12.6f}' * 3 + '\n' + ''.join(f'{m:5d}' + ''.join(f'{b * (i == j):12.6f}' for j in range(3)) + '\n' for i in range(3)) + ''.join(f'{z[i]:5d}{float(z[i]):12.6f}' + ''.join(f'{c / 0.529177210903:12.6f}' for c in atoms[i]) + '\n' for i in range(4))
    raw = (head + ''.join(f'{v_:13.5E}' + '\n' * ((i + 1) % 6 == 0) for i, v_ in enumerate(field.ravel()))).encode(); f2, st, zz, pp = read_cube(raw)
    assert list(zz) == z and np.allclose(pp * 0.529177210903, atoms, atol=1e-5) and np.allclose(st, b)
    net = Network(('A',), {'A': 4, 'B': 2}, {'A': -1 / 3, 'B': -0.8}, 1.62, 2.2, 1.17, site_radius=1.0, shell=(1.5, 2.5), core_radius=0.5, field_smoothing=0.1)
    site, shell = potentials(raw, sym, net); a0, a1 = site_and_shell(f2, st, pp, 1.0, (1.5, 2.5), 0.5, 0.1)
    assert np.allclose(site, reference_to_species(a0 * HARTREE, sym)) and np.allclose(shell, reference_to_species(a1 * HARTREE, sym)) and abs(site).max() > 0.05
    depth, conf = well(site, shell); assert np.allclose(depth, -site) and np.allclose(conf, shell - site)


def test_mulliken_takes_the_last_block_and_shares_count_magnitude():
    from sagw.spin import mulliken_spin
    block = lambda s: '\n'.join([' Mulliken Population Analysis', ' #  Atom  Element  Kind  Atomic population (alpha,beta) Net charge  Spin moment'] + [f'{i + 1:8d}      A{1:8d}  {1.9:10.6f}  {1.8:10.6f}  {0.3:10.6f}  {x:10.6f}' for i, x in enumerate(s)] + [' # Total charge and spin  0.0  1.0'])
    assert np.allclose(mulliken_spin(block([0.1, 0.9]) + '\n' + block([0.7, 0.3]))[1], [0.7, 0.3])
    w, n = shares(np.array([0.5, -0.5])); assert np.allclose(w, 0.5) and abs(n - 2) < 1e-12
    s = np.array([0.5, 0.3, 0.1, 0.05, 0.05]); w, n_eff = shares(s); k = int(np.ceil(n_eff)); assert carries(s, k - 1)[0] and not carries(s, k)[0] and carries(s, k - 1)[1] == k


def test_kernel_regression_and_normal_scores():
    from sklearn.metrics.pairwise import rbf_kernel
    rng = np.random.default_rng(6); X = rng.normal(size=(50, 4)); y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=50); Z, V = prepare(X[:40], X[40:]); a = 0.3; p = Z.shape[1]
    K, Q = rbf_kernel(Z, Z, gamma=1 / p), rbf_kernel(V, Z, gamma=1 / p); A = np.linalg.inv(K + a * np.eye(40)); one = np.ones(40); mean = one @ A @ y[:40] / (one @ A @ one)
    assert np.abs(smoother(Z, V, 'rbf', a) @ y[:40] - (Q @ A @ (y[:40] - mean) + mean)).max() < 1e-10 and np.abs(smoother(Z, V, 'rbf', a) @ np.full(40, 3.0) - 3).max() < 1e-10
    with pytest.raises(ValueError, match='ridge or rbf'): smoother(Z, V, 'rfb', a)
    tr = np.array([[1.], [2.], [2.], [3.]]); te = np.array([[2.], [9.]]); g, h = prepare(tr, te, gaussianise=True); assert abs(g[1, 0] - g[2, 0]) < 1e-12 and abs(h[0, 0] - g[1, 0]) < 1e-12 and h[1, 0] == g[3, 0]
    sc = norm.ppf((np.arange(1, 6) - .5) / 5); g, _ = prepare(np.array([[1.], [2.], [3.], [4.], [10.]]), np.zeros((1, 1)), gaussianise=True); assert np.allclose(g[:, 0], (sc - sc.mean()) / sc.std(), rtol=0, atol=1e-12)


def test_neighbourhoods_by_bonds():
    _, sym, cell, pos = cell_zero(); t = int(pd.read_csv(DATA / 'events.csv').trap_atom[0]) - 1; r = event_descriptors(pos[0], pos[1], cell, sym, t, NET, radii='bonds')
    u = displacement(pos[0], pos[1], cell); hop = bonds_from(contact_graph(distances(pos[0], cell), NET.role_of(sym), NET), t)
    assert len(r) == 215 and all(abs(r[f'response__within{h}bonds__squared_displacement_A2'] - (u[hop <= h] ** 2).sum()) < 1e-12 for h in (1, 2, 3))
    with pytest.raises(ValueError, match='bonds'): event_descriptors(pos[0], pos[1], cell, sym, t, NET, radii='bond')


def test_the_generator_rebuilds_the_first_example_cell():
    import scipy
    from sagw.structures import read_extxyz
    made = ' '.join((DATA / 'README.md').read_text().split())
    if f'NumPy {np.__version__}' not in made or f'SciPy {scipy.__version__}' not in made: pytest.skip('the example data were made with other NumPy and SciPy versions')
    sys.path.insert(0, str(ROOT / 'example')); import make_example as M
    rng = np.random.default_rng(M.SEED); kinds = [k for k in M.CLASSES for _ in range(M.PER_CLASS)]; rng.shuffle(kinds); sym, pos, charged, cell, trap = M.draw(rng, kinds[0])
    s0, p0, c0 = read_extxyz(DATA / 'structures/q0/cell_001.extxyz'); s1, p1, _ = read_extxyz(DATA / 'structures/q-1/cell_001.extxyz')
    assert (s0 == sym).all() and np.abs(p0 - pos).max() < 1e-7 and np.abs(p1 - charged).max() < 1e-7 and np.allclose(c0, cell) and trap == pd.read_csv(DATA / 'events.csv').trap_atom[0] - 1
