"""Share of the added electron on every atom, from a spin-density grid or Mulliken populations; relocation check."""
import re
import numpy as np
from scipy.spatial import cKDTree

_MULLIKEN = re.compile(r'^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
SHARE_FLOOR = 5e-7               # half the last printed digit of a Mulliken spin; smaller shares are raised to this before the logarithm


def mulliken_spin(text, n_atoms=None):
    """Mulliken net charge and spin of every atom from the last spin-resolved Mulliken population block (index, element, kind, population alpha, beta, net charge, spin)."""
    lines = text.splitlines(); starts = [i for i, l in enumerate(lines) if 'Mulliken Population Analysis' in l]
    if not starts:
        raise ValueError('no Mulliken block')
    rows = []
    for l in lines[starts[-1] + 1:]:
        hit = _MULLIKEN.match(l)
        if hit: rows.append((float(hit.group(6)), float(hit.group(7))))
        elif rows or 'Total charge' in l: break
    if not rows:
        raise ValueError('the last Mulliken block is not spin resolved')
    if n_atoms is not None and len(rows) != n_atoms:
        raise ValueError(f'the last Mulliken block holds {len(rows)} atoms, not {n_atoms}')
    q, s = np.array(rows).T; return q, s


def voronoi_integral(field, step, pos):
    """Integral of a grid field over the Voronoi cell of every atom."""
    shape = field.shape; lengths = np.array(shape) * step; wrapped = np.mod(pos, lengths); wrapped[wrapped >= lengths] = 0.0
    grid = np.stack(np.meshgrid(*[np.arange(shape[k]) * step[k] for k in range(3)], indexing='ij'), axis=-1).reshape(-1, 3)
    owner = cKDTree(wrapped, boxsize=lengths).query(grid, k=1, workers=1)[1]
    return np.bincount(owner, weights=field.ravel(), minlength=len(pos)) * float(np.prod(step))


def shares(spin):
    """Share of the added electron on every atom, from |spin|, and the effective number of atoms that carry it (inverse participation ratio).  The spin
    is the added electron when the neutral state is closed shell; negative spin counts too."""
    w = np.abs(np.asarray(spin, float))
    if not w.sum() > 0:
        raise ValueError('no spin on any atom: use the spin of the charged cell')
    w = w / w.sum(); return w, float(1 / (w ** 2).sum())


def log_share(w):
    """Logarithm of a share of the added electron, floored at SHARE_FLOOR."""
    return np.log(np.maximum(w, SHARE_FLOOR))


def carries(spin, atom):
    """Whether an atom is among the N_eff atoms with the largest share; also its rank, its share and N_eff."""
    w, n_eff = shares(spin); rank = int((w > w[atom]).sum() + 1); return rank <= np.ceil(n_eff), rank, float(w[atom]), n_eff
