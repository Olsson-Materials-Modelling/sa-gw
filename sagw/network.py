"""Settings of a network: species roles and coordinations, contact cutoff and edge weight, scales, and the radii of the reductions; from the bond length or measured from structures."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

RATIOS = dict(cutoff=1.35, sigma=0.72, spheres=(1.73, 2.31, 2.89), site_radius=0.58, shell=(0.81, 1.73))   # per bond length: contact cutoff, weight width, descriptor spheres, site radius, shell of the potential


@dataclass
class Network:
    role_a: tuple                                              # species that bond to the other role; contacts are pairs of unlike role
    expected: dict                                             # coordination of each species
    ideal_cosine: dict                                         # cosine of the ideal bond angle at each species
    bond: float                                                # bond length (A)
    cutoff: float                                              # contact cutoff (A)
    sigma: float                                               # width of the Gaussian edge weight (A)
    scales: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)                  # diffusion times of the wavelets
    spheres: tuple = None                                      # descriptor neighbourhood radii (A)
    site_radius: float = None                                  # potential at the site: within this radius of the nucleus (A)
    shell: tuple = None                                        # potential over the shell: between these radii (A)

    def __post_init__(self):
        self.role_a, self.scales = tuple(self.role_a), tuple(self.scales)
        self.spheres = tuple(self.spheres) if self.spheres else tuple(r * self.bond for r in RATIOS['spheres'])
        self.site_radius = self.site_radius or RATIOS['site_radius'] * self.bond
        self.shell = tuple(self.shell) if self.shell else tuple(r * self.bond for r in RATIOS['shell'])

    @classmethod
    def from_bond_length(cls, role_a, expected, ideal_cosine, bond, **kw):
        """Cutoff and weight width from the bond length by RATIOS."""
        return cls(role_a, expected, ideal_cosine, bond, RATIOS['cutoff'] * bond, RATIOS['sigma'] * bond, **kw)

    @classmethod
    def from_structures(cls, role_a, expected, ideal_cosine, symbols, positions, cells, **kw):
        """Bond length and cutoff measured from cells: the first peak of the unlike-pair distances and the middle of the first gap after it."""
        from .structures import distances
        d = []
        for s, p, c in zip(symbols, positions, cells):
            role = np.isin(s, role_a); dd = distances(p, c); d.append(dd[np.triu(role[:, None] != role[None, :], 1)])
        d = np.concatenate(d); d = d[d < 2.5 * d.min()]; counts, edges = np.histogram(d, bins=120); mids = 0.5 * (edges[1:] + edges[:-1])
        peak = int(np.argmax(counts)); bond = float(mids[peak]); upto = int(np.searchsorted(mids, 1.8 * bond)); low = np.flatnonzero(counts[peak:upto] == counts[peak:upto].min()) + peak
        return cls(role_a, expected, ideal_cosine, bond, float(mids[low].mean()), RATIOS['sigma'] * bond, **kw)

    def role_of(self, symbols):
        """True for atoms of role A."""
        return np.isin(symbols, self.role_a)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1) + '\n')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
