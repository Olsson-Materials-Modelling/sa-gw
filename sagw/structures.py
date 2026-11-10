"""Structure files, minimum-image geometry and displacement fields; orthorhombic cells, Angstrom."""
import re
from collections import Counter
from pathlib import Path
import numpy as np


def read_extxyz(path):
    """Symbols, positions (A) and cell lengths (A) of one extended-XYZ file; the columns from its Properties (species:S:1:pos:R:3 when it has none)."""
    lines = Path(path).read_text().splitlines(); n = int(lines[0]); lat = np.array(re.search(r'Lattice="([^"]*)"', lines[1]).group(1).split(), float).reshape(3, 3)
    if np.abs(lat - np.diag(np.diag(lat))).max() > 1e-9:
        raise ValueError(f'{path}: the cell is not orthorhombic')
    prop = re.search(r'Properties=(\S+)', lines[1]); f = (prop.group(1) if prop else 'species:S:1:pos:R:3').split(':'); col, at = {}, 0
    for name, width in zip(f[::3], f[2::3]): col[name] = at; at += int(width)
    if 'species' not in col or 'pos' not in col:
        raise ValueError(f'{path}: Properties names no species or no pos')
    rows = [l.split() for l in lines[2:2 + n]]; s, x = col['species'], col['pos']
    return np.array([r[s] for r in rows]), np.array([[float(v) for v in r[x:x + 3]] for r in rows]), np.diag(lat).copy()


def read_structure_files(folder):
    """Cell names, symbols, cell lengths and positions (cells, 2, atoms, 3) of the neutral (q0) and charged (q-1) structures; every cell must have the
    same number of atoms."""
    folder = Path(folder); names = sorted(p.stem for p in (folder / 'q0').glob('*.extxyz'))
    if names != sorted(p.stem for p in (folder / 'q-1').glob('*.extxyz')):
        raise ValueError('q0 and q-1 do not hold the same cells')
    sym, pos, cell = [], [], []
    for name in names:
        s0, p0, c0 = read_extxyz(folder / 'q0' / f'{name}.extxyz'); s1, p1, c1 = read_extxyz(folder / 'q-1' / f'{name}.extxyz')
        if len(s0) != len(s1) or not (s0 == s1).all() or np.abs(c1 - c0).max() > 1e-9:
            raise ValueError(f'{name}: the two endpoints differ in atoms or cell')
        sym.append(s0); pos.append(np.stack([p0, p1])); cell.append(c0)
    size = Counter(map(len, sym)).most_common(1)[0][0] if sym else 0
    if any(len(s) != size for s in sym):
        raise ValueError(f'cells of other than {size} atoms: ' + ', '.join(f'{n} ({len(s)})' for n, s in zip(names, sym) if len(s) != size))
    return names, np.array(sym), np.array(cell), np.array(pos)


def minimum_image(v, cell):
    """Vectors folded into the nearest periodic image."""
    return v - np.rint(v / cell) * cell


def separations(pos, cell):
    """Minimum-image vectors between all atoms of one cell."""
    return minimum_image(pos[None] - pos[:, None], cell)


def distances(pos, cell):
    """Minimum-image distances between all atoms of one cell."""
    return np.linalg.norm(separations(pos, cell), axis=2)


def displacement(pos0, pos1, cell):
    """Displacement from the neutral to the charged structure, minimum image, mean translation removed; cell lengths of an orthorhombic cell."""
    cell = np.asarray(cell, float)
    if cell.shape != (3,):
        raise ValueError('give the three cell lengths of an orthorhombic cell')
    u = minimum_image(pos1 - pos0, cell)
    if np.abs(u).max() >= cell.min() / 4:
        raise ValueError('displacements reach a quarter of the cell: register the two endpoints first')
    return u - u.mean(0)
