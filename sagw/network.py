"""Network settings: A species, coordinations, bond angles, cutoff, edge weight, scales and field-reduction radii; from a bond length or measured
from structures."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from .potential import SIGMA, R_CORE

RATIOS = dict(cutoff=1.35, sigma=0.72, spheres=(1.73, 2.31, 2.89), site_radius=0.58, shell=(0.81, 1.73))   # per bond length: bond cutoff, weight width, descriptor spheres, site radius, shell radii of the potential


def _round(x):
    return round(float(x), 6)


@dataclass
class Network:
    role_a: tuple                                              # A species; all others are B, and a bond is an A-B pair
    expected: dict                                             # expected coordination of each species
    ideal_cosine: dict                                         # cosine of the ideal bond angle at each species
    bond: float                                                # bond length (A)
    cutoff: float                                              # bond cutoff (A)
    sigma: float                                               # 1/e radius of the Gaussian edge weight exp(-(d/sigma)^2) (A)
    scales: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)                  # diffusion times of the wavelets
    spheres: tuple = None                                      # radii of the descriptor neighbourhoods (A)
    site_radius: float = None                                  # site potential: average within this radius of the nucleus (A)
    shell: tuple = None                                        # shell potential: average between these radii (A)
    smooth: float = 0.0                                        # half-width of a smooth cutoff (A); 0 is a sharp cutoff
    reference: dict = None                                     # species whose potentials are referenced to the mean of another species (an impurity to its host)
    field_smoothing: float = SIGMA                             # standard deviation of the Gaussian smoothing of a potential grid (A)
    core_radius: float = R_CORE                                # shell points closer than this to a nucleus are left out (A)

    def __post_init__(self):
        self.role_a, self.scales = tuple(self.role_a), tuple(self.scales)
        self.bond, self.cutoff, self.sigma, self.smooth, self.field_smoothing, self.core_radius = map(_round, (self.bond, self.cutoff, self.sigma, self.smooth, self.field_smoothing, self.core_radius))
        self.spheres = tuple(map(_round, self.spheres if self.spheres else [r * self.bond for r in RATIOS['spheres']]))
        self.site_radius = _round(self.site_radius if self.site_radius is not None else RATIOS['site_radius'] * self.bond)
        self.shell = tuple(map(_round, self.shell if self.shell else [r * self.bond for r in RATIOS['shell']]))
        self.reference = dict(self.reference or {})
        if self.smooth < 0:
            raise ValueError(f'smooth {self.smooth} A: the half-width of the switch cannot be negative')
        if any(int(k) != k or k < 1 for k in self.expected.values()) or set(self.expected) - set(self.ideal_cosine):
            raise ValueError('every species needs a whole expected coordination of at least 1 and an ideal cosine')
        self.expected = {s: int(k) for s, k in self.expected.items()}

    @classmethod
    def from_bond_length(cls, role_a, expected, ideal_cosine, bond, **kw):
        """Cutoff and weight width from the bond length by RATIOS."""
        return cls(role_a, expected, ideal_cosine, bond, RATIOS['cutoff'] * bond, RATIOS['sigma'] * bond, **kw)

    @classmethod
    def from_structures(cls, role_a, expected, ideal_cosine, symbols, positions, cells, **kw):
        """Bond length and cutoff measured from neutral cells: the bond length is the most common distance from an atom to its k nearest A-B neighbours
        (k: expected coordination), the cutoff the middle of the gap before the next.  Refuses a cutoff at which the mean coordination is off by more than 10 %."""
        from .structures import distances
        ranked, first, after, unlike = [], [], [], []
        for s, p, c in zip(symbols, positions, cells):
            s = np.asarray(s); role = np.isin(s, role_a); k = np.array([int(expected[x]) for x in s]); hd = np.where(role[:, None] != role[None, :], distances(p, c), np.inf); r = np.sort(hd, 1)[:, :2 * k.max() + 2]
            ranked.append((r, k)); first.append(r[np.arange(r.shape[1]) < k[:, None]]); after.append(r[np.arange(len(r)), k]); unlike.append(hd[np.triu(np.isfinite(hd), 1)])
        first, after, unlike = map(np.concatenate, (first, after, unlike)); edges = np.linspace(first.min(), 2.5 * first.min(), 121); mids = 0.5 * (edges[1:] + edges[:-1])
        peak, nxt = (int(np.argmax(np.histogram(x, edges)[0])) for x in (first, after)); counts = np.histogram(unlike, edges)[0]
        if nxt <= peak:
            raise ValueError('the next nearest unlike atoms do not lie beyond the first shell')
        low = np.flatnonzero(counts[peak:nxt + 1] == counts[peak:nxt + 1].min()) + peak; bond, cutoff = float(mids[peak]), float(mids[low].mean())
        z = sum(int((r <= cutoff).sum()) for r, _ in ranked) / sum(int(k.sum()) for _, k in ranked)
        if abs(z - 1) > 0.1:
            raise ValueError(f'mean coordination at the cutoff {cutoff:.3f} A is {z:.2f} of the expected: check the histogram of the unlike distances')
        return cls(role_a, expected, ideal_cosine, bond, cutoff, RATIOS['sigma'] * bond, **kw)

    def role_of(self, symbols):
        """True for A atoms."""
        return np.isin(symbols, self.role_a)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
