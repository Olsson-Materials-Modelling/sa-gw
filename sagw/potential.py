"""Electrostatic potential at the site and over the shell of every atom from a Hartree-potential grid (Gaussian cube)."""
import numpy as np
import scipy.fft as sfft

BOHR = 0.529177210903; HARTREE = 27.211386245988
SIGMA = 0.15                     # Gaussian smoothing of the grid field (A)
R_CORE = 0.6                     # shell points closer than this to another nucleus are left out (A)


def read_cube(raw):
    """Gaussian cube bytes -> field, grid spacing (Bohr), atomic numbers, positions (Bohr); orthorhombic grid from the origin."""
    head = raw[:60000].split(b'\n'); nat = int(head[2].split()[0]); origin = np.array(head[2].split()[1:4], float)
    n = [int(head[3 + i].split()[0]) for i in range(3)]; step = np.array([head[3 + i].split()[1:4] for i in range(3)], float)
    atoms = np.array([l.split() for l in head[6:6 + abs(nat)]], float); skip = 6 + abs(nat) + (1 if nat < 0 else 0)
    values = np.array(raw[sum(len(l) + 1 for l in head[:skip]):].split(), float)
    if values.size != n[0] * n[1] * n[2]:
        raise ValueError(f'cube size {values.size} != {n}')
    if np.abs(origin).max() > 1e-9 or np.abs(step - np.diag(np.diag(step))).max() > 1e-9:
        raise ValueError('non-orthorhombic or shifted cube grid')
    return values.reshape(n), np.diag(step).copy(), atoms[:, 0].astype(int), atoms[:, 2:5]


def smooth(field, step):
    """Gaussian smoothing of width SIGMA on the grid."""
    spec = sfft.fftn(field)
    for i in range(3):
        k = 2 * np.pi * sfft.fftfreq(field.shape[i], d=step[i] * BOHR); spec *= np.exp(-0.5 * (SIGMA * k) ** 2).reshape([-1 if j == i else 1 for j in range(3)])
    return sfft.ifftn(spec).real


def site_and_shell(field, step, pos, site_radius, shell):
    """Site and shell averages of a grid field around every atom: within site_radius of the nucleus and between the shell radii (A)."""
    R_SITE, R_SHELL = site_radius, tuple(shell)
    fine = smooth(field, step); shape = np.array(field.shape); h = step * BOHR; lengths = shape * h; p = np.mod(pos * BOHR, lengths); n = len(p); base = np.rint(p / h).astype(int); flat = fine.ravel()
    if (lengths <= R_SHELL[1] + R_CORE).any():
        raise ValueError(f'cell {lengths} too small for a shell of {R_SHELL[1]} A')

    def offsets(radius):
        reach = [int(np.ceil(radius / h[k])) + 1 for k in range(3)]; return np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing='ij'), axis=-1).reshape(-1, 3)
    off = offsets(R_SITE); site = np.empty(n)
    for a in range(n):
        idx = base[a] + off; inside = np.linalg.norm(idx * h - p[a], axis=1) < R_SITE; site[a] = flat[np.ravel_multi_index(np.mod(idx[inside], shape).T, tuple(shape))].mean()
    off = offsets(R_CORE); clear = np.ones(flat.size, bool)                                 # grid points farther than R_CORE from every nucleus
    for a in range(n):
        idx = base[a] + off; clear[np.ravel_multi_index(np.mod(idx[np.linalg.norm(idx * h - p[a], axis=1) <= R_CORE], shape).T, tuple(shape))] = False
    off = offsets(R_SHELL[1]); d = np.linalg.norm(off * h, axis=1); pad = np.linalg.norm(h); off = off[(d >= R_SHELL[0] - pad) & (d < R_SHELL[1] + pad)]; shell = np.empty(n)
    for a in range(n):
        idx = base[a] + off; r = np.linalg.norm(idx * h - p[a], axis=1); f = np.ravel_multi_index(np.mod(idx[(r >= R_SHELL[0]) & (r < R_SHELL[1])], shape).T, tuple(shape)); shell[a] = flat[f[clear[f]]].mean()
    return site, shell


def reference_to_species(values, symbols):
    """Values relative to the mean of the same species in the cell."""
    out = np.array(values, float)
    for s in np.unique(symbols): out[symbols == s] -= out[symbols == s].mean()
    return out


def potentials(cube_bytes, symbols, net):
    """Site and shell potential energy of an electron at every atom (eV) from a v_hartree cube, with the radii of the network settings."""
    field, step, _, pos = read_cube(cube_bytes); site, shell = site_and_shell(field, step, pos, net.site_radius, net.shell)
    return reference_to_species(site * HARTREE, symbols), reference_to_species(shell * HARTREE, symbols)


def well(site, shell):
    """Depth and confinement of the well at an atom."""
    return -np.asarray(site), np.asarray(shell) - np.asarray(site)
