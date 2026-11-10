"""Fingerprints of every example event in the neutral and charged structure, and their summary values (demo/out/events.csv, fingerprint.npz)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files, distances, displacement
from sagw.graph import contact_graph, contact_switch, graph_weights, laplacian, operator, scattering, bonds_from, structural_channels, standardise, readings, STRUCTURE
from sagw.network import Network
from sagw.spin import log_share

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'
CHANNELS = STRUCTURE + ['site_potential_eV', 'well_contrast_eV', 'electron_share', 'log_electron_share']
FAMILY = {'structure': slice(0, 7), 'energetic': slice(7, 9), 'electronic': slice(9, 11)}


def main(out=OUT):
    net = Network.load(DATA / 'network.json'); scales = net.scales; names, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); a = np.load(DATA / 'potentials.npz')
    for label, order in (('events.csv', list(ev.cell)), ('potentials.npz', list(a['cells']))):
        if order != names:
            raise ValueError(f'{label} does not list the cells of structures/ in their order')
    nb, n = len(names), sym.shape[1]; role = net.role_of(sym); trap = ev.trap_atom.to_numpy() - 1; site, shell = a['site_eV'], a['shell_eV']
    e = pd.read_parquet(DATA / 'electron.parquet'); e['b'] = e.cell.map({x: i for i, x in enumerate(names)})
    share = np.stack([np.full((nb, n), np.nan)] * 2, 1)
    for q, st in enumerate(('P', 'T')):
        t = e[e.state == st]; share[t.b.to_numpy(), q, t.atom.to_numpy() - 1] = t.share.to_numpy()
    gap = np.isnan(share).any(axis=(1, 2))
    if gap.any():
        raise ValueError('cells without a share of the electron on every atom: ' + ', '.join(np.array(names)[gap]))
    F = np.empty((nb, 2, n, len(CHANNELS))); F[..., :7] = [[structural_channels(pos[b, q], cell[b], sym[b], net) for q in range(2)] for b in range(nb)]
    F[..., 7] = site; F[..., 8] = shell - site; F[..., 9] = share; F[..., 10] = log_share(share); Z = standardise(F, role)
    S = np.empty((nb, 2, n, len(scales), len(CHANNELS)), np.float32); hop = np.zeros((nb, n), np.int16); rows = []
    for b in range(nb):
        d = [distances(pos[b, q], cell[b]) for q in range(2)]; contact = [contact_graph(d[q], role[b], net) for q in range(2)]; w = [graph_weights(d[q], contact_switch(d[q], role[b], net), net) for q in range(2)]
        try:
            H = [operator(w[q]) for q in range(2)]
        except ValueError as err:
            raise ValueError(f'{names[b]}: {err}') from None
        V = [scattering(H[q], Z[b, q], scales) for q in range(2)]; S[b] = V
        j = trap[b]; hop[b] = bonds_from(contact[0], j); u = displacement(pos[b, 0], pos[b, 1], cell[b]); mag = np.linalg.norm(u, axis=1)
        L, deg = laplacian(w[0]); lam, U = np.linalg.eigh(L); root = np.sqrt(deg)
        radius = {t: float(np.sqrt(((((U[j] * np.exp(-t * lam)) @ U.T) * root / root[j]) * d[0][j] ** 2).sum())) for t in scales}           # reach of H_t from the trap atom (A)
        rec = dict(cell=names[b], contacts_before=int(contact[0][j].sum()), contacts_after=int(contact[1][j].sum()), graph_deformation_A2=float(.5 * (w[0] * ((u[:, None] - u[None]) ** 2).sum(2)).sum()), max_displacement_A=float(mag.max()))
        rec.update({f'scale_radius_t{t:g}_A': radius[t] for t in scales}); rec.update({f'displacement_power_t{t:g}_A2': float(((root[:, None] * (H[0](t, u) - H[0](2 * t, u))) ** 2).sum()) for t in scales})
        rec.update({f'shell_displacement_h{h}_A': float(np.median(mag[hop[b] == h])) if (hop[b] == h).any() else np.nan for h in range(6)})
        for fam, cs in FAMILY.items():
            t0, eta, eta_t = readings(V[0][j, :, cs], V[1][j, :, cs]); rec[f'T0__{fam}'] = t0; rec[f'eta__{fam}'] = eta; rec.update({f'eta_t{t:g}__{fam}': float(x) for t, x in zip(scales, eta_t)})
            dv = np.linalg.norm((V[1] - V[0])[..., cs].reshape(n, -1), axis=1); rec.update({f'reach_h{h}__{fam}': float(np.median(dv[hop[b] == h])) if (hop[b] == h).any() else np.nan for h in range(6)})
        rows.append(rec)
    out.mkdir(exist_ok=True); ev.merge(pd.DataFrame(rows), on='cell').to_csv(out / 'events.csv', index=False)
    np.savez_compressed(out / 'fingerprint.npz', S=S, Z=Z.astype(np.float32), F=F, channels=np.array(CHANNELS), scales=np.array(scales), cells=np.array(names), symbols=sym, trap=trap, kind=ev.kind.to_numpy(str), hop=hop)


if __name__ == '__main__':
    main()
