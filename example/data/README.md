# Example data

100 random network glasses of 119 atoms (a four-fold network former A bridged by two-fold B, 0.066 atoms per A3, bond length
1.62 A), relaxed with a model energy, each paired with the relaxed structure after one trapping event.  Synthetic throughout;
the generator is `example/make_example.py` (deterministic).  Thirty cells per class are generated and the twenty nearest the
class median of the fingerprint readings are kept.

    network.json                   the settings every script reads: roles, coordinations, bond angles, bond length, cutoff, weight width, scales, radii
    structures/q0/<cell>.extxyz    the neutral cell;  structures/q-1/<cell>.extxyz  the relaxed cell with the trapped electron
    events.csv                     cell, kind, trap_atom, precursor_atom (1-based), level_before_eV, level_after_eV, stabilisation_eV
    potentials.npz                 cells; site_eV, shell_eV (cells x 2 endpoints x atoms): the electrostatic potential energy of an
                                   electron at the site (within the site radius of the nucleus) and over the shell, relative to the species mean
    electron.parquet               cell, state (P before, T after relaxation), atom, species, share: share of the added electron

Five kinds of event, 20 cells each (levels in eV below the conduction band edge; gap 8 eV):

    dangling bond    the most regular three-coordinate A atom; empty level in the upper half of the gap (-2.5), occupied level in the
                     lower half (-5.6); the electron on that atom before and after; a deep, strongly confined well that trapping
                     hardly changes; the atom pyramidalises and little else moves
    polaron          the most regular four-coordinate A atom; it keeps its contacts; the state starts at the band edge (-0.1),
                     extended over the cell, and localises on the atom and its four neighbours at mid gap (-4.0); a shallow well
                     deepens and spreads over the neighbours; every bond lengthens and the shells beyond breathe outwards
    bond break       starts as a polaron on the most strained four-coordinate site, whose longest bond is stretched; that bond breaks
                     and the B atom leaves; the state goes from the band edge to the dangling-bond level (-5.6) and localises on the
                     now three-coordinate atom; the well becomes deep and strongly confined
    substitutional   a larger impurity X on a regular A site (longer bonds) with a fixed shallow level (-1.3, shifting 0.3 on
                     trapping), the electron on the atom; a deep, unchanging well; the atom moves off centre towards its two nearest
                     neighbours
    interstitial     an impurity I bonded to two B atoms in a void, fixed deep level (-6.8, shifting 0.3), the electron on the atom;
                     the deepest well, unchanging; it binds a third B atom

An intrinsic occupied level deepens by 0.3 eV per e-fold of the graph deformation.  The stabilisation is 0.15 times the level
drop, less 0.05 eV per A2 of graph deformation, plus 0.05 eV and 0.02 eV of noise.  Event displacements are fixed per class; every
atom also moves by 0.01 A at random.  In four dangling-bond cells the electron first sits on the other dangling bond of the cell
(precursor_atom).

Potential: the potential at every atom's site follows its coordination deficit and bond length with noise; the trap atom
carries the depth and confinement of its class; the shell value is the site value plus the confinement.  Electron shares decay
with the number of bonds from the atom carrying the state, with the width of the class.
