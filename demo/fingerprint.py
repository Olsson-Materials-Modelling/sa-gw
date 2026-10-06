#!/usr/bin/env python3
"""The fingerprint of every example event at both endpoints and its readings (demo/out/events.csv, fingerprint.npz)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sagw.structures import read_structure_files, distances, displacement
from sagw.graph import contact_graph, graph_weights, operator, scattering, bonds_from, structural_channels, STRUCTURE
from sagw.network import Network

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'
CHANNELS = STRUCTURE + ['site_potential_eV', 'well_contrast_eV', 'electron_share', 'electron_reach']
FAMILY = {'structure': slice(0, 7), 'energetic': slice(7, 9), 'electronic': slice(9, 11)}


def main():
    net = Network.load(DATA / 'network.json'); SCALES = net.scales; boxes, sym, cell, pos = read_structure_files(DATA / 'structures'); ev = pd.read_csv(DATA / 'events.csv'); assert list(ev.cell) == boxes
    nb, n = len(boxes), sym.shape[1]; role = net.role_of(sym); trap = ev.trap_atom.to_numpy() - 1
    a = np.load(DATA / 'potentials.npz', allow_pickle=True); assert list(a['cells']) == boxes; site, shell = a['site_eV'], a['shell_eV']
    e = pd.read_parquet(DATA / 'electron.parquet'); e['b'] = e.cell.map({x: i for i, x in enumerate(boxes)})
    share = np.stack([np.full((nb, n), np.nan)] * 2, 1)
    for q, st in enumerate(('P', 'T')):
        t = e[e.state == st]; share[t.b.to_numpy(), q, t.atom.to_numpy() - 1] = t.share.to_numpy()
    Fs = np.stack([np.stack([structural_channels(pos[b, q], cell[b], sym[b], net) for q in range(2)]) for b in range(nb)])
    mu, sd = Fs[:, 0].reshape(-1, 7).mean(0), Fs[:, 0].reshape(-1, 7).std(0)
    F = np.empty((nb, 2, n, len(CHANNELS))); F[..., :7] = Fs; F[..., 7] = site; F[..., 8] = shell - site; F[..., 9] = share; F[..., 10] = np.log(share)
    Z = np.empty_like(F); Z[..., :7] = (Fs - mu) / sd; C = F[..., 7:].copy()
    for m in (role, ~role):                                                 # field channels: centred by role, one spread for both
        C[:, 0][m] -= F[:, 0][m][:, 7:].mean(0); C[:, 1][m] -= F[:, 0][m][:, 7:].mean(0)
    Z[..., 7:] = C / C[:, 0].reshape(-1, 4).std(0)
    S = np.empty((nb, 2, n, len(SCALES), len(CHANNELS)), np.float32); hop = np.zeros((nb, n), np.int16); rows = []
    for b in range(nb):
        d = [distances(pos[b, q], cell[b]) for q in range(2)]; contact = [contact_graph(d[q], role[b], net) for q in range(2)]; w = [graph_weights(d[q], contact[q], net) for q in range(2)]
        for q in range(2): S[b, q] = scattering(operator(w[q]), Z[b, q], SCALES)
        j = trap[b]; hop[b] = bonds_from(contact[0], j); u = displacement(pos[b, 0], pos[b, 1], cell[b]); mag = np.linalg.norm(u, axis=1); H = operator(w[0]); root = np.sqrt(w[0].sum(1))[:, None]
        lam, U = np.linalg.eigh(np.eye(n) - (w[0] / np.sqrt(np.outer(w[0].sum(1), w[0].sum(1))))); s_, r_ = 1 / np.sqrt(w[0].sum(1)), np.sqrt(w[0].sum(1))
        radius = {t: float(np.sqrt(((s_[j] * ((U[j] * np.exp(-t * lam)) @ U.T) * r_) * d[0][j] ** 2).sum())) for t in SCALES}           # reach of H_t from the trap atom, in A
        rec = dict(cell=boxes[b], contacts_before=int(contact[0][j].sum()), contacts_after=int(contact[1][j].sum()), graph_deformation_A2=float(.5 * (w[0] * ((u[:, None] - u[None]) ** 2).sum(2)).sum()),
                   max_displacement_A=float(mag.max()), displacement_participation_atoms=float((mag ** 2).sum() ** 2 / (mag ** 4).sum()), deficit_neutral=float(F[b, 0, j, 0]), deficit_anion=float(F[b, 1, j, 0]))
        rec.update({f'scale_radius_t{t:g}_A': radius[t] for t in SCALES}); rec.update({f'displacement_power_t{t:g}_A2': float(((root * (H(t, u) - H(2 * t, u))) ** 2).sum()) for t in SCALES}); rec.update({f'shell_displacement_h{h}_A': float(np.median(mag[hop[b] == h])) if (hop[b] == h).any() else np.nan for h in range(6)})
        for fam, cs in FAMILY.items():
            s0, s1 = S[b, 0, j, :, cs].astype(float), S[b, 1, j, :, cs].astype(float); r0, r1 = np.linalg.norm(s0, axis=1), np.linalg.norm(s1, axis=1)
            rec[f'T0__{fam}'] = float(r0.sum()); rec[f'Tminus__{fam}'] = float(r1.sum()); rec[f'eta__{fam}'] = float(np.linalg.norm(s1 - s0) / np.linalg.norm(s0))
            rec.update({f'eta_t{t:g}__{fam}': float(np.linalg.norm(s1[k] - s0[k]) / r0[k]) for k, t in enumerate(SCALES)})
            dv = np.linalg.norm((S[b, 1] - S[b, 0])[..., cs].astype(float).reshape(n, -1), axis=1); rec.update({f'reach_h{h}__{fam}': float(np.median(dv[hop[b] == h])) if (hop[b] == h).any() else np.nan for h in range(6)})
        rows.append(rec)
    out = ev.merge(pd.DataFrame(rows), on='cell'); OUT.mkdir(exist_ok=True); out.to_csv(OUT / 'events.csv', index=False)
    np.savez_compressed(OUT / 'fingerprint.npz', S=S, Z=Z.astype(np.float32), F=F, channels=np.array(CHANNELS), scales=np.array(SCALES), cells=np.array(boxes), symbols=sym, trap=trap, kind=ev.kind.to_numpy(), hop=hop)
    pd.set_option('display.width', 200); print(out.groupby('kind')[['T0__structure', 'eta__structure', 'eta__energetic', 'eta__electronic', 'graph_deformation_A2', 'contacts_before', 'contacts_after']].median().round(3).to_string())


if __name__ == '__main__':
    main()
