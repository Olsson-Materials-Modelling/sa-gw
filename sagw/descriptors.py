"""Descriptors of the structural response to one trapping event (response__, wavelet__, quadratic__), over the cell and over spheres around the trap atom."""
import numpy as np
from .structures import separations, displacement, minimum_image
from .graph import contact_graph, contact_switch, graph_weights, laplacian, bonds_from, structural_channels, STRUCTURE


def _participation(x):
    x = np.asarray(x, float); q = (x ** 2).sum(); return float(x.sum() ** 2 / q) if q > 1e-24 else np.nan


def _tensor(vec):
    """Trace, largest-eigenvalue fraction, anisotropy and isotropy of the second-moment tensor of a set of vectors."""
    if len(vec) == 0: return (np.nan,) * 4
    G = vec.T @ vec / len(vec); e = np.linalg.eigvalsh(G); tr = float(e.sum())
    if tr <= 1e-24: return (0., np.nan, np.nan, np.nan)
    return tr, float(e[-1] / tr), float((e[-1] - e[0]) / tr), float(np.linalg.det(G) / (tr / 3) ** 3)


def _geometry(pos, cell, role, net):
    v = separations(pos, cell); d = np.linalg.norm(v, axis=2); B = contact_graph(d, role, net); hd = np.where(role[:, None] != role[None, :], d, np.inf); order = np.argsort(hd, axis=1)[:, :6]
    return dict(v=v, d=d, B=B, z=B.sum(1), order=order, rd=np.take_along_axis(hd, order, 1))


def event_descriptors(pos0, pos1, cell, symbols, trap, net, radii=None):
    """Descriptors of one event from the neutral (pos0) and charged (pos1) positions in the same atom order; trap 0-based; radii in A (default: the
    network's spheres) or 'bonds' for neighbourhoods of one to three bonds."""
    if isinstance(radii, str) and radii != 'bonds':
        raise ValueError(f"radii {radii!r}: radii in A or 'bonds'")
    radii = net.spheres if radii is None else radii; role = net.role_of(symbols); g0, g1 = _geometry(pos0, cell, role, net), _geometry(pos1, cell, role, net); j = trap; row = {}
    if isinstance(radii, str):
        hop = bonds_from(g0['B'], j); hoods = {f'within{h}bonds': hop <= h for h in (1, 2, 3)}
    else:
        hoods = {f'sphere{k + 1}': g0['d'][j] <= r for k, r in enumerate(radii)}
    u = displacement(pos0, pos1, cell); u2 = (u ** 2).sum(1); R = 'response__'
    row.update({R + 'trap_atom_squared_displacement_A2': u2[j], R + 'cell_squared_displacement_A2': u2.sum(), R + 'cell_displacement_participation': _participation(u2), R + 'cell_largest_displacement_A': np.sqrt(u2.max()), R + 'cell_rms_displacement_A': np.sqrt(u2.mean())})
    a0, a1 = set(np.flatnonzero(g0['B'][j])), set(np.flatnonzero(g1['B'][j]))
    for lab, m in (('lost', a0 - a1), ('gained', a1 - a0), ('retained', a0 & a1)):                      # bonds of the trap atom lost, gained and kept
        k = np.array(sorted(m), int); val = g1['d'][j, k] - g0['d'][j, k]
        row[f'{R}trap_atom_{lab}_contacts'] = len(k); row[f'{R}trap_atom_{lab}_length_change_sum_A'] = val.sum(); row[f'{R}trap_atom_{lab}_length_change_sqsum_A2'] = (val ** 2).sum()
    for rank in range(1, 6):                                                                           # the trap atom's neutral neighbours, followed by index
        lig = int(g0['order'][j, rank - 1]); row[f'{R}neutral_ligand{rank}_distance_change_A'] = g1['d'][j, lig] - g0['d'][j, lig]
    bi, bj = np.where(np.triu(g0['B'], 1))
    for name, sh in hoods.items():
        others = sh.copy(); others[j] = False; tn, tc = (_tensor(g['v'][j, others] - g['v'][j, others].mean(0)) for g in (g0, g1))   # gyration of the neighbours about their centroid
        for k, lab in enumerate(('gyration_trace_A2', 'largest_axis_fraction', 'anisotropy', 'isotropy')): row[f'{R}{name}__{lab}_change'] = tc[k] - tn[k]
        q = u2[sh]; row.update({f'{R}{name}__squared_displacement_A2': q.sum(), f'{R}{name}__rms_displacement_A': np.sqrt(q.mean()), f'{R}{name}__largest_displacement_A': np.sqrt(q.max()), f'{R}{name}__displacement_participation': _participation(q),
                                f'{R}{name}__share_of_cell_squared_displacement': q.sum() / u2.sum(), f'{R}{name}__displacement_anisotropy': _tensor(u[sh])[2]})
        keep = sh[bi] | sh[bj]; ii, jj = bi[keep], bj[keep]; v0, v1 = g0['v'][ii, jj], g1['v'][ii, jj]; d0, d1 = g0['d'][ii, jj], g1['d'][ii, jj]   # neutral bonds touching the neighbourhood
        dv = minimum_image(v1 - v0, cell); lon = np.einsum('ij,ij->i', dv, v0 / d0[:, None]); dl = d1 - d0; mean0 = float(d0.mean())
        row.update({f'{R}{name}__bond_longitudinal_sqsum_A2': (lon ** 2).sum(), f'{R}{name}__bond_fractional_longitudinal_sqsum': ((lon / d0) ** 2).sum(), f'{R}{name}__bond_transverse_sqsum_A2': np.maximum(np.einsum('ij,ij->i', dv, dv) - lon ** 2, 0).sum(),
                    f'{R}{name}__bond_length_change_sqsum_A2': (dl ** 2).sum()})
        n_, c_ = ((d0 - mean0) ** 2).sum(), ((d1 - mean0) ** 2).sum(); row[f'{R}{name}__bond_length_spread_sqsum_A2_change'] = c_ - n_; row[f'{R}{name}__bond_length_spread_sqsum_A2_release'] = n_ - c_
        row[f'{R}{name}__bond_linear_recovery_A2'] = -2 * ((d0 - mean0) * dl).sum(); lost = ~g1['B'][ii, jj]
        row[f'{R}{name}__bonds_lost'] = lost.sum(); row[f'{R}{name}__lost_bond_extension_sqsum_A2'] = (dl[lost] ** 2).sum()
        c0, c1, ideal = [], [], []
        for ce in np.flatnonzero(sh):                                                                  # bond angles at the neighbourhood atoms, over their neutral bonds
            nb = np.flatnonzero(g0['B'][ce])
            if len(nb) < 2: continue
            x0 = g0['v'][ce, nb] / g0['d'][ce, nb, None]; x1 = g1['v'][ce, nb] / g1['d'][ce, nb, None]; ai, aj = np.triu_indices(len(nb), 1)
            c0.extend(np.einsum('ij,ij->i', x0[ai], x0[aj])); c1.extend(np.einsum('ij,ij->i', x1[ai], x1[aj])); ideal.extend([net.ideal_cosine[symbols[ce]]] * len(ai))
        c0, c1, ideal = map(np.asarray, (c0, c1, ideal))
        for lab, n_, c_ in (('angle_strain_sqsum', ((c0 - ideal) ** 2).sum(), ((c1 - ideal) ** 2).sum()), ('angle_strain_sqmean', ((c0 - ideal) ** 2).mean(), ((c1 - ideal) ** 2).mean())):
            row[f'{R}{name}__{lab}_change'] = c_ - n_; row[f'{R}{name}__{lab}_release'] = n_ - c_
        row[f'{R}{name}__angle_cosine_change_sqsum'] = ((c1 - c0) ** 2).sum(); row[f'{R}{name}__angle_linear_recovery'] = -2 * ((c0 - ideal) * (c1 - c0)).sum()
    X0, X1 = structural_channels(pos0, cell, symbols, net)[j], structural_channels(pos1, cell, symbols, net)[j]
    for k, ch in enumerate(STRUCTURE): row[f'{R}trap_atom_{ch}_change'] = X1[k] - X0[k]
    for r in range(6): row[f'{R}trap_atom_rank{r + 1}_distance_change_A'] = g1['rd'][j, r] - g0['rd'][j, r]
    row[R + 'trap_atom_contact_count_change'] = g1['z'][j] - g0['z'][j]
    # displacement wavelets p_it = d_i |((H_t - H_2t) u)_i|^2
    w = graph_weights(g0['d'], contact_switch(g0['d'], role, net), net); L, deg = laplacian(w); lam, U = np.linalg.eigh(L); x = np.sqrt(deg)[:, None] * u
    P = np.stack([((U @ ((np.exp(-t * lam) - np.exp(-2 * t * lam))[:, None] * (U.T @ x))) ** 2).sum(1) for t in net.scales]); areas = {'cell': np.ones(len(u), bool), **hoods}; Wv, Q = 'wavelet__', 'quadratic__'
    for name, m in areas.items():
        p = P[:, m]; sums = p.sum(1); den = (p * p).sum(1); neff = np.divide(sums * sums, den, out=np.full(len(net.scales), np.nan), where=den > 1e-30); tot = sums.sum(); fr = sums / tot
        for k, t in enumerate(net.scales): row[f'{Wv}{name}__power_t{t:g}_A2'] = sums[k]; row[f'{Wv}{name}__participation_t{t:g}'] = neff[k]; row[f'{Wv}{name}__share_t{t:g}'] = fr[k]
        row[f'{Wv}{name}__scale_entropy'] = -np.sum(fr * np.log(np.maximum(fr, 1e-300))); row[f'{Wv}{name}__power_all_scales_A2'] = tot; row[f'{Wv}{name}__mean_scale'] = np.exp(fr @ np.log(net.scales))
    an = P[:, j]; tot = an.sum(); fr = an / tot
    for k, t in enumerate(net.scales): row[f'{Wv}trap_atom__power_t{t:g}_A2'] = an[k]; row[f'{Wv}trap_atom__amplitude_t{t:g}_A'] = np.sqrt(an[k]); row[f'{Wv}trap_atom__share_t{t:g}'] = fr[k]
    row[f'{Wv}trap_atom__scale_entropy'] = -np.sum(fr * np.log(np.maximum(fr, 1e-300))); row[f'{Wv}trap_atom__power_all_scales_A2'] = tot; row[f'{Wv}trap_atom__mean_scale'] = np.exp(fr @ np.log(net.scales))
    ci, cj = np.nonzero(np.triu(w > 0, 1)); ww = w[ci, cj]; du = u[cj] - u[ci]; du2 = (du ** 2).sum(1); lon = np.einsum('ij,ij->i', du, g0['v'][ci, cj] / g0['d'][ci, cj, None])   # every pair with a weight: the bonds when smooth is 0
    for name, m in areas.items():
        e = m[ci] | m[cj]; D = float((ww[e] * du2[e]).sum()); Dl = float((ww[e] * lon[e] ** 2).sum())
        row[f'{Q}{name}__deformation_A2'] = D; row[f'{Q}{name}__longitudinal_deformation_A2'] = Dl; row[f'{Q}{name}__transverse_deformation_A2'] = max(D - Dl, 0.); row[f'{Q}{name}__longitudinal_share'] = Dl / D if D > 1e-30 else np.nan
    row[Q + 'cell__normalised_weight_deformation_A2'] = float(.5 * (w / np.sqrt(np.outer(deg, deg)) * ((u[:, None] - u[None]) ** 2).sum(2)).sum()); return row   # zero for a rigid translation
