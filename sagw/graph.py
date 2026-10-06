"""Contact graph of a two-role network, diffusion operator H_t = D^-1/2 exp(-tL) D^1/2, wavelets W_t = H_t - H_2t, scattering S_t = H_t |W_t x|, structural channels; the settings come from a Network."""
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .structures import separations

STRUCTURE = ['coord_deficit', 'contact_excess', 'bond_mean_A', 'bond_std_A', 'outer_rank_distance_A', 'next_rank_gap_A', 'angle_cos_rms']


def contact_graph(d, role, net):
    """Boolean contact matrix: pairs of unlike role within the cutoff."""
    het = role[:, None] != role[None, :]; return het & (d <= net.cutoff) & (d > 1e-12)


def graph_weights(d, contact, net):
    """Gaussian edge weights of the contact graph."""
    w = np.where(contact, np.exp(-(d / net.sigma) ** 2), 0.); np.fill_diagonal(w, 0); return w


def laplacian(w):
    """Symmetric normalised Laplacian and the degrees."""
    deg = w.sum(1); s = 1 / np.sqrt(deg); return np.eye(len(w)) - s[:, None] * w * s[None, :], deg


def operator(w):
    """H_t y = D^-1/2 exp(-tL) D^1/2 y, as a function of (t, y)."""
    L, deg = laplacian(w); s = 1 / np.sqrt(deg); r = np.sqrt(deg); lam, U = np.linalg.eigh(L)
    return lambda t, y: s[:, None] * (U @ (np.exp(-t * lam)[:, None] * (U.T @ (r[:, None] * y))))


def scattering(H, x, scales):
    """Scattering coefficients S_t = H_t |(H_t - H_2t) x|, shape (atoms, scales, channels)."""
    h = {t: H(t, x) for t in sorted(set(scales) | {2 * t for t in scales})}; return np.stack([H(t, np.abs(h[t] - h[2 * t])) for t in scales], 1)


def bonds_from(contact, atom, cap=6):
    """Bonds on the shortest path from one atom to every other, capped."""
    g = shortest_path(csr_matrix(contact), unweighted=True, indices=atom); return np.where(np.isfinite(g), np.minimum(g, cap), cap).astype(np.int16)


def structural_channels(pos, cell, symbols, net):
    """The seven structural channels of every atom of one cell."""
    v = separations(pos, cell); d = np.linalg.norm(v, axis=2); role = net.role_of(symbols); out = np.zeros((len(pos), 7))
    for i in range(len(pos)):
        opp = np.flatnonzero(role != role[i]); o = opp[np.argsort(d[i, opp])]; r = d[i, o]; k = net.expected[symbols[i]]; z = int((r <= net.cutoff).sum()); m = min(z, k)
        out[i, 0] = max(k - z, 0); out[i, 1] = max(z - k, 0); out[i, 4] = r[k - 1]; out[i, 5] = r[k] - r[k - 1]
        if m:
            out[i, 2] = r[:m].mean(); out[i, 3] = r[:m].std()
        if m >= 2:
            u = v[i, o[:m]] / r[:m, None]; c = np.array([u[a] @ u[b] for a, b in combinations(range(m), 2)]); out[i, 6] = np.sqrt(((c - net.ideal_cosine[symbols[i]]) ** 2).mean())
    return out
