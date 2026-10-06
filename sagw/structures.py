"""Structure files, minimum-image geometry and displacement fields; orthorhombic cells, Angstrom."""
import re
import numpy as np



def read_extxyz(path):
    """Symbols, positions (A) and cell lengths (A) of one extended-XYZ file."""
    lines = path.read_text().splitlines(); n = int(lines[0]); lat = np.array(re.search(r'Lattice="([^"]*)"', lines[1]).group(1).split(), float).reshape(3, 3)
    if np.abs(lat - np.diag(np.diag(lat))).max() > 1e-9:
        raise ValueError(f'{path}: the cell is not orthorhombic')
    rows = [l.split() for l in lines[2:2 + n]]; return np.array([r[0] for r in rows]), np.array([[float(v) for v in r[1:4]] for r in rows]), np.diag(lat).copy()


def read_structure_files(folder):
    """Names, symbols, cell lengths and positions of the paired q0 and q-1 cells; endpoint 0 neutral, 1 anion."""
    boxes = sorted(p.stem for p in (folder / 'q0').glob('*.extxyz'))
    if boxes != sorted(p.stem for p in (folder / 'q-1').glob('*.extxyz')):
        raise ValueError('q0 and q-1 do not hold the same cells')
    sym, pos, cell = [], [], []
    for box in boxes:
        s0, p0, c0 = read_extxyz(folder / 'q0' / f'{box}.extxyz'); s1, p1, c1 = read_extxyz(folder / 'q-1' / f'{box}.extxyz')
        if not (s0 == s1).all() or np.abs(c1 - c0).max() > 1e-9:
            raise ValueError(f'{box}: the two endpoints differ in atoms or cell')
        sym.append(s0); pos.append(np.stack([p0, p1])); cell.append(c0)
    return boxes, np.array(sym), np.array(cell), np.array(pos)


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
    """Displacement between two endpoints, minimum image, mean translation removed."""
    u = minimum_image(pos1 - pos0, cell)
    if np.abs(u).max() >= np.min(np.diag(cell) if np.ndim(cell) == 2 else cell) / 4:
        raise ValueError('displacements reach a quarter of the cell: register the two endpoints first')
    return u - u.mean(0)
