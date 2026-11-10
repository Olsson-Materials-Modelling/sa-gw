"""Guards on the inputs and the settings: Mulliken blocks, cube files, atoms without bonds, electron shares, referencing, structure folders, the
measured cutoff, the standardisation and the smooth cutoff.  pytest -q"""
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import sagw.structures
from sagw.structures import read_structure_files, distances, minimum_image, separations, displacement
from sagw.graph import contact_graph, contact_switch, graph_weights, laplacian, operator, structural_channels, standardise
from sagw.network import Network
from sagw.potential import read_cube, reference_to_species, site_and_shell
from sagw.models import prepare
from sagw.spin import mulliken_spin, log_share, SHARE_FLOOR
from sagw.sensitivity import sensitivity, structural_readings

DATA = ROOT / 'example/data'; NET = Network.load(DATA / 'network.json')


def mulliken(n, spin=True):
    head = ['', ' Mulliken Population Analysis', '', ' #  Atom  Element  Kind  Atomic population' + (' (alpha,beta) Net charge  Spin moment' if spin else '  Net charge')]
    rows = [f'{i + 1:8d}      A{1:8d}  {1.9:10.6f}  {1.8:10.6f}  {0.3:10.6f}  {0.1 * (i % 3):10.6f}' if spin else f'{i + 1:8d}      A{1:8d}  {3.7:10.6f}  {0.3:10.6f}' for i in range(n)]
    return '\n'.join(head + rows + [' # Total charge' + ('  and spin   0.000000   1.000000' if spin else '   0.000000'), ''])


def cube(nat, n=(4, 4, 4), scale=1.0):
    head = ['cube', 'v_hartree', f'{nat:5d}' + f'{0:12.6f}' * 3] + [f'{n[i]:5d}' + ''.join(f'{0.5 * (i == j):12.6f}' for j in range(3)) for i in range(3)]
    atoms = [f'{14:5d}{14:12.6f}{k % 7 * .3:12.6f}{k % 5 * .3:12.6f}{k % 3 * .3:12.6f}' for k in range(nat)]; v = np.linspace(-1, 1, int(np.prod(np.abs(n)))) * scale
    return ('\n'.join(head + atoms) + '\n' + ''.join(f'{x:13.5E}' + '\n' * ((i + 1) % 6 == 0) for i, x in enumerate(v)) + '\n').encode(), v


def step_channels(pos, cell, symbols, net):
    """The structural channels with the step at the cutoff, written out directly."""
    v = separations(pos, cell); d = np.linalg.norm(v, axis=2); role = net.role_of(symbols); out = np.zeros((len(pos), 7))
    for i in range(len(pos)):
        opp = np.flatnonzero(role != role[i]); o = opp[np.argsort(d[i, opp])]; r = d[i, o]; k = net.expected[symbols[i]]; z = int((r <= net.cutoff).sum()); m = min(z, k)
        out[i, 0] = max(k - z, 0); out[i, 1] = max(z - k, 0); out[i, 4] = r[k - 1]; out[i, 5] = r[k] - r[k - 1]
        if m: out[i, 2] = r[:m].mean(); out[i, 3] = r[:m].std()
        if m >= 2:
            u = v[i, o[:m]] / r[:m, None]; c = np.array([u[a] @ u[b] for a, b in combinations(range(m), 2)]); out[i, 6] = np.sqrt(((c - net.ideal_cosine[symbols[i]]) ** 2).mean())
    return out


def test_mulliken_reads_every_atom_of_a_large_cell():
    q, s = mulliken_spin(mulliken(1000), n_atoms=1000); assert len(s) == 1000 and abs(s[2] - 0.2) < 1e-12 and abs(q[0] - 0.3) < 1e-12
    with pytest.raises(ValueError, match='not 999'): mulliken_spin(mulliken(1000), n_atoms=999)
    with pytest.raises(ValueError, match='not spin resolved'): mulliken_spin(mulliken(10, spin=False))


def test_cube_reader_takes_large_headers_and_joined_fields():
    raw, v = cube(1500); field, step, z, pos = read_cube(raw); assert field.shape == (4, 4, 4) and len(z) == 1500 and np.allclose(step, 0.5) and np.allclose(field.ravel(), v, rtol=1e-5, atol=0)
    raw, v = cube(3, scale=1e-101); assert b'E-101-' in raw and np.allclose(read_cube(raw)[0].ravel(), v, rtol=1e-5, atol=0)
    with pytest.raises(ValueError, match='Bohr'): read_cube(cube(3, n=(-4, 4, 4))[0])
    lines = cube(2)[0].split(b'\n'); lines[2] = b'   -2' + lines[2][5:]; mo = b'\n'.join(lines[:8] + [b'    1  641'] + lines[8:]); field, _, z, pos = read_cube(mo)
    assert field.shape == (4, 4, 4) and list(z) == [14, 14] and np.allclose(pos[1], [0.3, 0.3, 0.3]) and np.allclose(field.ravel(), cube(2)[1], rtol=1e-5, atol=0)
    head = b'\n'.join(cube(1, n=(6, 1, 1))[0].split(b'\n')[:7]) + b'\n'                                                    # a header for six values; bodies as Fortran writes them
    assert np.allclose(read_cube(head + b'  0.10000-100 -0.25000-119  0.37000D+00 -0.15000E+01  0.30000E-04  0.10000+101\n')[0].ravel(), [1e-101, -2.5e-120, 0.37, -1.5, 3e-5, 1e100], rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match='not numbers'): read_cube(head + b'  0.1 0.2 0.3 0.4 0.5 abc\n')


def test_an_atom_without_contacts_is_refused():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); d = distances(pos[0, 0], cell[0]); w = graph_weights(d, contact_graph(d, NET.role_of(sym[0]), NET), NET); w[7] = w[:, 7] = 0
    for f in (laplacian, operator):
        with pytest.raises(ValueError, match=r'\[7\]'): f(w)


def test_a_zero_share_has_a_finite_logarithm():
    x = log_share(np.array([0., 1e-9, 0.5])); assert np.isfinite(x).all() and x[0] == x[1] == np.log(SHARE_FLOOR) and x[2] == np.log(0.5)


def test_referencing_keeps_a_single_impurity():
    sym = np.array(['A', 'A', 'A', 'B', 'B', 'X']); v = np.array([1., 2., 3., 5., 7., 10.])
    assert np.allclose(reference_to_species(v, sym, {'X': 'A'}), [-1, 0, 1, -1, 1, 8]) and np.allclose(reference_to_species(list(v[:5]), list(sym[:5])), [-1, 0, 1, -1, 1])
    with pytest.raises(ValueError, match='host'): reference_to_species(v, sym)
    with pytest.raises(ValueError, match='no A atom'): reference_to_species([1., 2., 3.], ['X', 'B', 'B'], {'X': 'A'})


def test_structure_folders_take_text_paths_and_name_cells_of_another_size(tmp_path):
    assert len(read_structure_files(str(DATA / 'structures'))[0]) == 100
    for q in ('q0', 'q-1'):
        (tmp_path / q).mkdir()
        for name in ('cell_001', 'cell_002', 'cell_003'):
            lines = (DATA / 'structures' / q / f'{name}.extxyz').read_text().splitlines(); n = int(lines[0]) - (name == 'cell_003')
            (tmp_path / q / f'{name}.extxyz').write_text('\n'.join([str(n), lines[1]] + lines[2:2 + n]) + '\n')
    with pytest.raises(ValueError, match=r'cell_003 \(118\)'): read_structure_files(tmp_path)


def test_the_cutoff_is_found_when_the_second_shell_is_taller(monkeypatch):
    rng = np.random.default_rng(5); na, nb = 40, 80; sym = np.array(['A'] * na + ['B'] * nb)

    def shells(pos, cell):
        d = np.full((na + nb, na + nb), 3.0); ab = rng.uniform(4.5, 6.0, (na, nb))
        for j in range(nb):
            a1, a2 = j // 2, (j // 2 + 1) % na; ab[[a1, a2], j] = rng.normal(1.74, 0.08, 2)
        for i in range(na):
            far = np.flatnonzero(ab[i] > 4); ab[i, rng.choice(far, 10, replace=False)] = rng.normal(2.95, 0.10, 10)
        d[:na, na:] = ab; d[na:, :na] = ab.T; np.fill_diagonal(d, 0); return d
    monkeypatch.setattr(sagw.structures, 'distances', shells)
    m = Network.from_structures(('A',), {'A': 4, 'B': 2}, {'A': -1 / 3, 'B': -0.8}, [sym] * 10, [None] * 10, [None] * 10); assert abs(m.bond - 1.74) < 0.05 and 2.0 < m.cutoff < 2.7


def test_standardisation_centres_every_channel_by_role():
    rng = np.random.default_rng(0); role = rng.random((6, 30)) < 0.4; F = rng.normal(size=(6, 2, 30, 3)) + 5 * role[:, None, :, None]; Z = standardise(F, role, shared=()); n0 = Z[:, 0]
    assert np.abs(n0[role].mean(0)).max() < 1e-12 and np.abs(n0[~role].mean(0)).max() < 1e-12 and np.allclose(n0.reshape(-1, 3).std(0), 1)
    mu = np.where(role[..., None], F[:, 0][role].mean(0), F[:, 0][~role].mean(0)); sd = (F[:, 0] - mu).reshape(-1, 3).std(0); assert np.allclose(Z[:, 1], (F[:, 1] - mu) / sd, rtol=0, atol=1e-12)
    mu[..., 0] = F[:, 0, :, 0].mean(); sd = (F[:, 0] - mu).reshape(-1, 3).std(0); assert np.allclose(standardise(F, role, shared=(0,)), (F - mu[:, None]) / sd, rtol=0, atol=1e-12)
    flat = np.zeros((2, 2, 6, 2)); flat[..., 1] = 1.0; role = np.tile([True, False, False, True, False, False], (2, 1)); assert np.abs(standardise(flat, role)).max() == 0   # full coordination of A and B stays at zero


def test_the_switch_reduces_to_the_step_and_is_continuous_across_the_cutoff():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); p0, c, s = pos[0, 0], cell[0], sym[0]; role = NET.role_of(s)
    assert np.abs(structural_channels(p0, c, s, NET) - step_channels(p0, c, s, NET)).max() < 1e-12
    j = int(np.flatnonzero(role)[0]); m = int(np.flatnonzero(contact_graph(distances(p0, c), role, NET)[j])[0]); u = minimum_image(p0[m] - p0[j], c); u /= np.linalg.norm(u)

    def at(r, net):
        p = p0.copy(); p[m] = p0[j] + r * u; d = distances(p, c); return structural_channels(p, c, s, net), graph_weights(d, contact_switch(d, role, net), net)
    with pytest.raises(ValueError, match='negative'): replace(NET, smooth=-0.05)
    soft = replace(NET, smooth=0.05); (a, wa), (b, wb) = at(NET.cutoff - 1e-6, soft), at(NET.cutoff + 1e-6, soft); (ha, _), (hb, _) = at(NET.cutoff - 1e-6, NET), at(NET.cutoff + 1e-6, NET)
    assert np.abs(a - b).max() < 1e-4 and np.abs(wa - wb).max() < 1e-4 and abs((hb - ha)[j, 0] - (hb - ha)[j, 1] - 1) < 1e-12


def test_sensitivity_at_unit_factor_reproduces_the_readings():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); t = pd.read_csv(DATA / 'events.csv').trap_atom.to_numpy()[:10] - 1
    summary, flips = sensitivity(sym[:10], pos[:10], cell[:10], t, NET, factors=(1.0,)); assert (summary.contact_flips == 0).all() and (summary.lone_atom_events == 0).all() and np.allclose(summary.filter(like='rho__'), 1) and flips.shape == (10, 2)
    short = structural_readings(sym[:2], pos[:2], cell[:2], t[:2], replace(NET, cutoff=1.0)); assert short.lone_atom.all() and short.T0.isna().all()


def test_auc_is_the_rank_statistic():
    sys.path.insert(0, str(ROOT / 'demo')); from analyse import auc
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1); y = rng.integers(0, 2, 60); s = np.round(rng.normal(size=60), 1); assert abs(auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_degenerate_neighbourhoods_and_settings_are_named():
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); s = sym[0].copy(); s[0] = 'Q'
    with pytest.raises(ValueError, match='Q'): structural_channels(pos[0, 0], cell[0], s, NET)
    few = np.array(['A'] + ['B'] * 4); p = np.array([[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0], [0, 0, 1.6], [-1.6, 0, 0]], float) + 5
    with pytest.raises(ValueError, match='atom 0 .A. has 4 unlike atoms'): structural_channels(p, np.full(3, 12.), few, NET)
    for bad in ({'A': 0, 'B': 2, 'X': 4, 'I': 2}, {'A': 4.5, 'B': 2, 'X': 4, 'I': 2}):
        with pytest.raises(ValueError, match='expected coordination'): replace(NET, expected=bad)
    with pytest.raises(ValueError, match='three cell lengths'): displacement(pos[0, 0], pos[0, 1], np.diag(cell[0]))


def test_copies_of_a_column_count_once_and_a_shell_must_fit_the_cell():
    rng = np.random.default_rng(2); x = rng.normal(size=(400, 3)); X = np.column_stack([x, 3 * x[:, 0] + 1 + 1e-12 * rng.normal(size=400), -x[:, 1]]); tr, te = prepare(X[:300], X[300:])
    assert tr.shape[1] == 3 and te.shape[1] == 3
    grid, step, p3 = np.full((24, 24, 24), 2.5), np.full(3, 6.0 / 24 / 0.529177210903), np.array([[1., 1., 1.]])
    with pytest.raises(ValueError, match='own image'): site_and_shell(grid, step, p3, 1.0, (1.4, 3.2))
    assert np.allclose(site_and_shell(grid, step, p3, 1.0, (1.4, 2.9), core=0.3, width=0.1), 2.5)


def test_the_switch_plateaus_and_lone_atoms_stay_out_of_rho(monkeypatch):
    from sagw.graph import switch
    soft = replace(NET, smooth=0.05); assert np.allclose(switch(np.array([NET.cutoff - 0.05, NET.cutoff, NET.cutoff + 0.05, NET.cutoff - 0.2, NET.cutoff + 0.2]), soft), [1, 0.5, 0, 1, 0])
    names, sym, cell, pos = read_structure_files(DATA / 'structures'); t = pd.read_csv(DATA / 'events.csv').trap_atom.to_numpy()[:12] - 1
    summary, _ = sensitivity(sym[:12], pos[:12], cell[:12], t, NET, factors=(0.8,)); cut = summary[summary.setting == 'cutoff'].iloc[0]
    assert 0 < cut.lone_atom_events < 12 and np.isfinite(cut.filter(like='rho__').to_numpy(float)).all()


def test_a_cutoff_that_misses_the_coordination_is_refused(monkeypatch):
    rng = np.random.default_rng(5); na, nb = 40, 80; sym = np.array(['A'] * na + ['B'] * nb)

    def shells(pos, cell):
        d = np.full((na + nb, na + nb), 3.0); ab = rng.uniform(4.5, 6.0, (na, nb))
        for j in range(nb):
            ab[[j // 2, (j // 2 + 1) % na], j] = rng.normal(1.74, 0.03, 2)
        for i in range(na):
            far = np.flatnonzero(ab[i] > 4); ab[i, rng.choice(far, 10, replace=False)] = rng.normal(2.95, 0.10, 10)
        d[:na, na:] = ab; d[na:, :na] = ab.T; np.fill_diagonal(d, 0); return d
    monkeypatch.setattr(sagw.structures, 'distances', shells)
    with pytest.raises(ValueError, match='mean coordination'): Network.from_structures(('A',), {'A': 5, 'B': 2}, {'A': -1 / 3, 'B': -0.8}, [sym] * 10, [None] * 10, [None] * 10)
