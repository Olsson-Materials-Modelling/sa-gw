"""Robustness of the structural fingerprint to the bond cutoff and the edge-weight width."""
from dataclasses import replace
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from .structures import distances
from .graph import contact_graph, contact_switch, graph_weights, operator, scattering, structural_channels, standardise, readings


def structural_readings(symbols, pos, cell, trap, net):
    """Per event: the bonds of the trap atom in the neutral and charged structure, whether any atom has no bond, and T0, eta, eta_t of the structural
    channels (NaN when an atom has no bond); symbols (cells, atoms), pos (cells, 2, atoms, 3), trap 0-based."""
    role = net.role_of(symbols); Z = standardise(np.stack([[structural_channels(pos[b, q], cell[b], symbols[b], net) for q in range(2)] for b in range(len(trap))]), role); rows = []
    for b, j in enumerate(trap):
        d = [distances(pos[b, q], cell[b]) for q in range(2)]; w = [graph_weights(d[q], contact_switch(d[q], role[b], net), net) for q in range(2)]; lone = any((x.sum(1) <= 0).any() for x in w)
        t0, eta, eta_t = (np.nan, np.nan, np.full(len(net.scales), np.nan)) if lone else readings(*[scattering(operator(w[q]), Z[b, q], net.scales)[j] for q in range(2)])
        rows.append(dict(contacts_before=int(contact_graph(d[0], role[b], net)[j].sum()), contacts_after=int(contact_graph(d[1], role[b], net)[j].sum()), lone_atom=lone, T0=t0, eta=eta, **{f'eta_t{t:g}': e for t, e in zip(net.scales, eta_t)}))
    return pd.DataFrame(rows)


def sensitivity(symbols, pos, cell, trap, net, factors=(0.95, 1.05)):
    """The cutoff and the weight width, each scaled by every factor.  Per setting: the events whose trap atom gains or loses a bond, the events with an
    atom left without a bond, and the Spearman rho of each summary value against the unscaled settings, over the events valid in both."""
    base = structural_readings(symbols, pos, cell, trap, net); rows, flips = [], {}
    for name in ('cutoff', 'sigma'):
        for f in factors:
            alt = structural_readings(symbols, pos, cell, trap, replace(net, **{name: f * getattr(net, name)})); key = f'{name}_x{f:g}'; both = ~(base.lone_atom | alt.lone_atom)
            flips[key] = (alt[['contacts_before', 'contacts_after']] != base[['contacts_before', 'contacts_after']]).any(axis=1).to_numpy()
            rows.append(dict(setting=name, factor=f, value_A=round(f * getattr(net, name), 6), contact_flips=int(flips[key].sum()), lone_atom_events=int(alt.lone_atom.sum()),
                             **{f'rho__{c}': float(spearmanr(base[c][both], alt[c][both])[0]) for c in base.columns[3:]}))
    return pd.DataFrame(rows), pd.DataFrame(flips)
