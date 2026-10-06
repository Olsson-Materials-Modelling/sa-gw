"""Share of the added electron on every atom from a spin-density grid or Mulliken populations; relocation."""
import re
import numpy as np
from scipy.spatial import cKDTree

_MULLIKEN = re.compile(r'^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')


def mulliken_spin(text):
    """Mulliken net charge and spin of every atom from a Mulliken population block (index, element, kind, population alpha, beta, net charge, spin)."""
    lines = text.splitlines(); starts = [i for i, l in enumerate(lines) if 'Mulliken Population Analysis' in l]
    if not starts:
        raise ValueError('no Mulliken block')
    rows = []
    for l in lines[starts[-1] + 1:starts[-1] + 400]:
        hit = _MULLIKEN.match(l)
        if hit: rows.append((float(hit.group(6)), float(hit.group(7))))
        elif rows: break
    q, s = np.array(rows).T; return q, s


def voronoi_integral(field, step, pos):
    """Integral of a grid field over the Voronoi cell of every atom."""
    shape = field.shape; lengths = np.array(shape) * step; wrapped = np.mod(pos, lengths); wrapped[wrapped >= lengths] = 0.0
    grid = np.stack(np.meshgrid(*[np.arange(shape[k]) * step[k] for k in range(3)], indexing='ij'), axis=-1).reshape(-1, 3)
    owner = cKDTree(wrapped, boxsize=lengths).query(grid, k=1, workers=1)[1]
    return np.bincount(owner, weights=field.ravel(), minlength=len(pos)) * float(np.prod(step))

def shares(spin):
    """Share of the added electron on every atom and the number of atoms that carry it."""
    w = np.abs(spin); w = w / w.sum(); return w, float(1 / (w ** 2).sum())


def carries(spin, atom):
    """Whether an atom is among the N_eff atoms of largest share, with rank, share and N_eff."""
    w, n_eff = shares(spin); rank = int((w > w[atom]).sum() + 1); return rank <= np.ceil(n_eff), rank, float(w[atom]), n_eff
