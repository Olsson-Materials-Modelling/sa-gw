"""The demo's numbers against the committed reference in tests/reference/, and the numbers the tutorial quotes against the same reference.  pytest -q
    python3 tests/test_reference.py      rewrites the reference after a deliberate change"""
import re, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo'))
import fingerprint, analyse
from sagw.network import Network
from sagw.sensitivity import structural_readings
from sagw.structures import read_structure_files

REF = Path(__file__).resolve().parent / 'reference'
EXACT, FITTED, COUNT = (1e-6, 1e-9), (0, 0.01), (0, 1)     # (relative, absolute) tolerances: linear algebra; accuracies and areas of iterative fits; counts of a fit
TABLES = {'events.csv': EXACT, 'event_descriptors.csv': EXACT, 'response_signature.csv': EXACT, 'model.csv': EXACT, 'scale_radius.csv': EXACT, 'sensitivity.csv': EXACT,
          'identification.csv': FITTED, 'class_separation.csv': FITTED, 'confusion.csv': COUNT, 'relocation.csv': (0, 0)}


def outputs(out):
    """The reference tables from a fresh run of the demo into out."""
    fingerprint.main(out); analyse.main(out); return {name: pd.read_csv(out / name) for name in TABLES}


@pytest.fixture(scope='module')
def fresh(tmp_path_factory):
    return outputs(tmp_path_factory.mktemp('out'))


@pytest.mark.parametrize('name', list(TABLES))
def test_the_numbers_match_the_reference(fresh, name):
    ref, new, (rtol, atol) = pd.read_csv(REF / name), fresh[name], TABLES[name]; assert list(new.columns) == list(ref.columns) and len(new) == len(ref)
    for c in ref.columns:
        if ref[c].dtype.kind in 'if' and new[c].dtype.kind in 'if':
            np.testing.assert_allclose(new[c], ref[c], rtol=rtol, atol=atol, equal_nan=True, err_msg=f'{name}: {c}')
        else:
            assert (new[c].fillna('').astype(str) == ref[c].fillna('').astype(str)).all(), f'{name}: {c}'


def test_the_sensitivity_readings_are_the_fingerprint_readings():
    names, sym, cell, pos = read_structure_files(ROOT / 'example/data/structures'); ref = pd.read_csv(REF / 'events.csv'); ev = pd.read_csv(ROOT / 'example/data/events.csv')
    r = structural_readings(sym, pos, cell, ev.trap_atom.to_numpy() - 1, Network.load(ROOT / 'example/data/network.json'))
    assert (r.contacts_before == ref.contacts_before).all() and (r.contacts_after == ref.contacts_after).all() and np.allclose(r.T0, ref.T0__structure, rtol=1e-10, atol=0) and np.allclose(r.eta, ref.eta__structure, rtol=1e-10, atol=0)


def test_the_tutorial_quotes_the_reference():
    text = ' '.join((ROOT / 'docs/tutorial.md').read_text().split()); model = pd.read_csv(REF / 'model.csv').r2.round(2).tolist(); rad = pd.read_csv(REF / 'scale_radius.csv').radius_median_A
    sens = pd.read_csv(REF / 'sensitivity.csv').set_index(['setting', 'factor']); bond = float(re.search(r'"bond": (\d+\.\d+)', (ROOT / 'example/data/network.json').read_text()).group(1))
    assert [float(x) for x in re.search(r'Held-out R² of the stabilisation energy: (\d+\.\d+) from the descriptors, (\d+\.\d+) from Q_D alone and (\d+\.\d+) from Q_D with the trap type', text).groups()] == model
    assert [float(x) for x in re.search(r'scale_radius\.csv`: (\d+\.\d+) to (\d+\.\d+) bond lengths for t = 0\.5 to 8', text).groups()] == [round(rad.iloc[0] / bond, 1), round(rad.iloc[-1] / bond, 1)]
    f1, f2, n, r1, r2 = re.search(r'changes the bond count of the trap atom in (\d+) and (\d+) of the (\d+) events; Spearman ρ of every summary value against the unscaled settings stays at or above (\d+\.\d+)\. Scaling σ by the same factors keeps it at or above (\d+\.\d+)', text).groups()
    low = lambda s: np.floor(sens.loc[s].filter(like='rho__').to_numpy().min() * 100) / 100
    assert [int(f1), int(f2), int(n)] == [sens.loc[('cutoff', 0.95), 'contact_flips'], sens.loc[('cutoff', 1.05), 'contact_flips'], len(pd.read_csv(REF / 'events.csv'))] and [float(r1), float(r2)] == [low('cutoff'), low('sigma')]


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        REF.mkdir(exist_ok=True)
        for name, table in outputs(Path(tmp)).items(): table.to_csv(REF / name, index=False)
