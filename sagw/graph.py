"""Bond graph of an A-B network, diffusion operator H_t = D^-1/2 exp(-tL) D^1/2, wavelets W_t = H_t - H_2t, scattering S_t = H_t |W_t x|,
structural channels, their standardisation and the summary values at the trap atom."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .structures import separations

STRUCTURE = ['coord_deficit', 'contact_excess', 'bond_mean_A', 'bond_std_A', 'outer_rank_distance_A', 'next_rank_gap_A', 'angle_cos_rms']
COUNTS = (0, 1)                  # deficit and excess: zero is full coordination for A and B alike


def contact_graph(d, role, net):
    """Bond matrix: A-B pairs closer than the cutoff."""
    het = role[:, None] != role[None, :]; return het & (d <= net.cutoff) & (d > 1e-12)


def switch(d, net):
    """Bond weight against distance: 1 below cutoff - smooth, 0 above cutoff + smooth, a cosine taper in between (1/2 at the cutoff); a sharp cutoff when smooth is 0."""
    if not net.smooth: return (d <= net.cutoff).astype(float)
    return 0.5 - 0.5 * np.sin(0.5 * np.pi * np.clip((d - net.cutoff) / net.smooth, -1, 1))


def contact_switch(d, role, net):
    """Bond weight of every A-B pair; 0 for A-A and B-B pairs."""
    return np.where((role[:, None] != role[None, :]) & (d > 1e-12), switch(d, net), 0.)


def graph_weights(d, contact, net):
    """Gaussian edge weights exp(-(d/sigma)^2) on the bonds, given as a boolean bond matrix or as bond weights (contact_switch)."""
    w = np.exp(-(d / net.sigma) ** 2) * contact; np.fill_diagonal(w, 0); return w


def laplacian(w):
    """Symmetric normalised Laplacian and the degrees; every atom needs at least one bond."""
    deg = w.sum(1); lone = np.flatnonzero(deg <= 0)
    if lone.size:
        raise ValueError(f'atoms without a contact (0-based): {lone.tolist()}')
    s = 1 / np.sqrt(deg); return np.eye(len(w)) - s[:, None] * w * s[None, :], deg


def operator(w):
    """Diffusion operator H_t y = D^-1/2 exp(-tL) D^1/2 y as a function of (t, y).  It preserves constants and the degree-weighted sum of y, not the plain sum."""
    L, deg = laplacian(w); s = 1 / np.sqrt(deg); r = np.sqrt(deg); lam, U = np.linalg.eigh(L)
    return lambda t, y: s[:, None] * (U @ (np.exp(-t * lam)[:, None] * (U.T @ (r[:, None] * y))))


def scattering(H, x, scales):
    """Scattering coefficients S_t = H_t |(H_t - H_2t) x|, shape (atoms, scales, channels).  The dyadic wavelets add up to H_t0 - H_2tmax: modes slower
    than diffusion time 2 tmax are not seen."""
    h = {t: H(t, x) for t in sorted(set(scales) | {2 * t for t in scales})}; return np.stack([H(t, np.abs(h[t] - h[2 * t])) for t in scales], 1)


def bonds_from(contact, atom, cap=6):
    """Number of bonds on the shortest path from one atom to every other, capped at cap."""
    g = shortest_path(csr_matrix(contact), unweighted=True, indices=atom); return np.where(np.isfinite(g), np.minimum(g, cap), cap).astype(np.int16)


def structural_channels(pos, cell, symbols, net):
    """The seven structural channels of every atom: coordination deficit and excess, mean and standard deviation of the bond lengths, distance to
    the k-th nearest A-B neighbour and the gap to the next, and the rms deviation of the bond-angle cosines from the ideal (k: expected coordination)."""
    missing = set(np.unique(symbols)) - set(net.expected)
    if missing:
        raise ValueError(f'species without an expected coordination in the settings: {sorted(missing)}')
    v = separations(pos, cell); d = np.linalg.norm(v, axis=2); role = net.role_of(symbols); out = np.zeros((len(pos), 7))
    for i in range(len(pos)):
        opp = np.flatnonzero(role != role[i]); o = opp[np.argsort(d[i, opp])]; r = d[i, o]; k = net.expected[symbols[i]]
        if len(r) <= k:
            raise ValueError(f'atom {i} ({symbols[i]}) has {len(r)} unlike atoms in the cell: the channels need more than its expected coordination of {k}')
        f = switch(r, net); z = f.sum(); a = f[:k]; n = a.sum()
        out[i, 0] = max(k - z, 0); out[i, 1] = max(z - k, 0); out[i, 4] = r[k - 1]; out[i, 5] = r[k] - r[k - 1]
        if n > 0:
            out[i, 2] = (a * r[:k]).sum() / n; out[i, 3] = np.sqrt((a * (r[:k] - out[i, 2]) ** 2).sum() / n)
        p, q = np.triu_indices(k, 1); ab = a[p] * a[q]
        if ab.sum() > 0:
            u = v[i, o[:k]] / r[:k, None]; c = np.einsum('ij,ij->i', u[p], u[q]); out[i, 6] = np.sqrt((ab * (c - net.ideal_cosine[symbols[i]]) ** 2).sum() / ab.sum())
    return out


def standardise(F, role, shared=COUNTS):
    """Channels centred separately on A and B atoms (the shared channels on all atoms) and scaled by one standard deviation per channel, from the
    neutral structures; F (cells, 2, atoms, channels), role (cells, atoms) True for A.  Separate centring keeps an A-B offset out of the finest wavelets."""
    Z = np.array(F, float); n0 = Z[:, 0]; pooled = n0.reshape(-1, Z.shape[-1]).mean(0); centre = [n0[m].mean(0) for m in (role, ~role)]
    for m, mu in zip((role, ~role), centre):
        mu[list(shared)] = pooled[list(shared)]
        for q in range(2): Z[:, q][m] -= mu
    sd = Z[:, 0].reshape(-1, Z.shape[-1]).std(0); return Z / np.where(sd > 1e-12, sd, 1.)


def readings(V0, V1):
    """Summary values of the fingerprint V (scales, channels) at the trap atom in the neutral and charged structure: T0 = sum_t |V0_t|,
    eta = |V- - V0| / |V0| and eta_t = |V-_t - V0_t| / |V0_t|."""
    r0 = np.linalg.norm(V0, axis=1); return float(r0.sum()), float(np.linalg.norm(V1 - V0) / np.linalg.norm(V0)), np.linalg.norm(V1 - V0, axis=1) / r0
