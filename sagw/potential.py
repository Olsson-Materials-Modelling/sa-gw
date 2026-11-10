"""Potential energy of an electron at every atom, averaged at the site and over a shell, from a Hartree-potential cube."""
import re
import numpy as np
import scipy.fft as sfft

BOHR = 0.529177210903; HARTREE = 27.211386245988
SIGMA = 0.15                     # standard deviation of the Gaussian smoothing of a grid field (A)
R_CORE = 0.6                     # shell points closer than this to a nucleus are left out (A)


def read_cube(raw):
    """Gaussian cube bytes -> field, grid spacing (Bohr), atomic numbers, positions (Bohr); orthorhombic grid from the origin, voxel counts positive (Bohr)."""
    nat = int(raw.split(b'\n', 3)[2].split()[0]); skip = 6 + abs(nat) + (1 if nat < 0 else 0); head = raw.split(b'\n', skip); body = head.pop()
    origin = np.array(head[2].split()[1:4], float); n = [int(head[3 + i].split()[0]) for i in range(3)]; step = np.array([head[3 + i].split()[1:4] for i in range(3)], float)
    if min(n) <= 0:
        raise ValueError(f'cube voxel counts {n}: only grids in Bohr (positive counts) are read')
    atoms = np.array([l.split() for l in head[6:6 + abs(nat)]], float)
    try:
        values = np.array(body.split(), float)
    except ValueError:                                                                     # Fortran output: a three-digit exponent loses its E, or fills the field and runs into the next one
        body = re.sub(rb'([Ee][-+]?\d+)(?=-)', rb'\1 ', re.sub(rb'(\d)([-+]\d{3})(?!\d)', rb'\1E\2', body.replace(b'D', b'E').replace(b'd', b'e')))
        try:
            values = np.array(body.split(), float)
        except ValueError:
            raise ValueError('cube values that are not numbers') from None
    if values.size != n[0] * n[1] * n[2]:
        raise ValueError(f'cube size {values.size} != {n}')
    if np.abs(origin).max() > 1e-9 or np.abs(step - np.diag(np.diag(step))).max() > 1e-9:
        raise ValueError('non-orthorhombic or shifted cube grid')
    return values.reshape(n), np.diag(step).copy(), atoms[:, 0].astype(int), atoms[:, 2:5]


def gaussian_smooth(field, step, width=SIGMA):
    """Gaussian smoothing of a periodic grid field, standard deviation width (A)."""
    spec = sfft.fftn(field)
    for i in range(3):
        k = 2 * np.pi * sfft.fftfreq(field.shape[i], d=step[i] * BOHR); spec *= np.exp(-0.5 * (width * k) ** 2).reshape([-1 if j == i else 1 for j in range(3)])
    return sfft.ifftn(spec).real


def site_and_shell(field, step, pos, site_radius, shell, core=R_CORE, width=SIGMA):
    """Site and shell averages of a smoothed grid field around every atom: within site_radius of the nucleus, and between the shell radii on grid points
    farther than core from every nucleus (radii in A).  The site average keeps the nucleus."""
    r_in, r_out = shell
    fine = gaussian_smooth(field, step, width); shape = np.array(field.shape); h = step * BOHR; lengths = shape * h; p = np.mod(pos * BOHR, lengths); n = len(p); base = np.rint(p / h).astype(int); flat = fine.ravel()
    if (lengths <= 2 * r_out).any():
        raise ValueError(f'cell {lengths} too small for a shell of {r_out} A: the shell would meet its own image')

    def offsets(radius):
        reach = [int(np.ceil(radius / h[k])) + 1 for k in range(3)]; return np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing='ij'), axis=-1).reshape(-1, 3)
    off = offsets(site_radius); site_mean = np.empty(n)
    for a in range(n):
        idx = base[a] + off; inside = np.linalg.norm(idx * h - p[a], axis=1) < site_radius; site_mean[a] = flat[np.ravel_multi_index(np.mod(idx[inside], shape).T, tuple(shape))].mean()
    off = offsets(core); clear = np.ones(flat.size, bool)                                   # grid points farther than core from every nucleus
    for a in range(n):
        idx = base[a] + off; clear[np.ravel_multi_index(np.mod(idx[np.linalg.norm(idx * h - p[a], axis=1) <= core], shape).T, tuple(shape))] = False
    off = offsets(r_out); d = np.linalg.norm(off * h, axis=1); pad = np.linalg.norm(h); off = off[(d >= r_in - pad) & (d < r_out + pad)]; shell_mean = np.empty(n)
    for a in range(n):
        idx = base[a] + off; r = np.linalg.norm(idx * h - p[a], axis=1); f = np.ravel_multi_index(np.mod(idx[(r >= r_in) & (r < r_out)], shape).T, tuple(shape)); shell_mean[a] = flat[f[clear[f]]].mean()
    return site_mean, shell_mean


def reference_to_species(values, symbols, reference=None):
    """Values relative to the mean of their species in the cell, or of the species that reference maps it to (an impurity to its host); a species
    referenced to itself needs two atoms."""
    v = np.asarray(values, float); symbols = np.asarray(symbols); reference = reference or {}; out = v.copy()
    for s in np.unique(symbols):
        host = reference.get(s, s); pool = symbols == host
        if host != s and not pool.any():
            raise ValueError(f'no {host} atom in the cell to reference {s} to')
        if pool.sum() < 2 and host == s:
            raise ValueError(f'one {s} atom in the cell: map {s} to a host species (Network.reference)')
        out[symbols == s] -= v[pool].mean()
    return out


def potentials(cube_bytes, symbols, net):
    """Site and shell potential energy of an electron at every atom (eV) from a V_HARTREE cube, with the radii and referencing of the settings.
    
    Use cubes of the neutral charge state at both geometries: the neutral cell, and a neutral single point at the relaxed q-1 geometry.  A q-1 cube
    contains the Hartree potential of the added electron itself.  The sign is that of CP2K's V_HARTREE, an electron's potential energy, the negative of
    the electrostatic potential; negate a cube with the physical sign."""
    field, step, _, pos = read_cube(cube_bytes); site, shell = site_and_shell(field, step, pos, net.site_radius, net.shell, net.core_radius, net.field_smoothing)
    return reference_to_species(site * HARTREE, symbols, net.reference), reference_to_species(shell * HARTREE, symbols, net.reference)


def well(site, shell):
    """Well depth and confinement at every atom: -site and shell - site."""
    return -np.asarray(site), np.asarray(shell) - np.asarray(site)
