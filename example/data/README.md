# Example data

A synthetic toy, not a real material: 100 cells of 119 atoms, a random network of four-coordinate A bridged by two-coordinate B
(0.066 atoms per Å³, bond length 1.62 Å), relaxed with a model energy.  Each cell holds one trapping event; the charged structure is
the neutral one moved by a fixed rule for its trap type.  17 % of the A atoms are under-coordinated.

`example/make_example.py` makes twenty cells per trap type and keeps them all; it is deterministic for given NumPy and SciPy versions
(these files: Python 3.13, NumPy 2.5.4, SciPy 1.18.1).

    network.json                   settings
    structures/q0/<cell>.extxyz    neutral structure;  structures/q-1/<cell>.extxyz  charged structure
    events.csv                     cell, kind (trap type), trap_atom, precursor_atom (1-based), level_before_eV, level_after_eV, stabilisation_eV
    potentials.npz                 cells; site_eV, shell_eV (cells x 2 x atoms), relative to the species mean (X and I: the A mean)
    electron.parquet               cell, state (P before, T after trapping), atom, species, share of the added electron

Trap types; levels are E − E_CB in eV, gap 8 eV:

    type             trap atom   bonds   level before -> after   relaxation
    dangling bond    A           3 -> 3  −2.5 -> −5.6            the atom pyramidalises
    polaron          A           4 -> 4  −0.15 -> −4.0           all bonds lengthen, shells beyond move outwards
    bond break       A           4 -> 3  −0.3 -> −5.6            the longest bond breaks, the atom pyramidalises
    substitutional   X           4 -> 4  −1.3, 0.3 deeper        the atom moves off centre
    interstitial     I           2 -> 3  −6.8, 0.3 deeper        the atom bonds a third B

Rules:

- Level: an intrinsic level deepens by 0.3 eV per e-fold of graph deformation beyond the typical value for its type.
- Stabilisation energy: 0.15 × level drop − 0.05 eV per Å² of graph deformation + 0.05 eV, with 0.02 eV of noise.
- Every atom also moves by 0.01 Å of noise per component.
- In four dangling-bond cells the electron starts on another three-coordinate A atom (precursor_atom).
- Potential: follows the coordination deficit, with noise; the trap atom takes the well of its type; shell = site + confinement.
- Share: decays with bonds from the atom carrying the electron, which has the largest share; written to six decimals, so distant
  atoms read 0.
