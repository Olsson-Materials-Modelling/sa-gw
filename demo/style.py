"""House style of the demo figures: Okabe-Ito palette, boxed axes with inward ticks, framed legends, panel letters."""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

OKABE_ITO = {'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9', 'bluishgreen': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999'}
KNOWN = ['dangling bond', 'bond break', 'polaron', 'substitutional', 'interstitial']    # trap types of the example data, in display order
PALETTE = [OKABE_ITO[k] for k in ('vermillion', 'blue', 'bluishgreen', 'purple', 'orange', 'skyblue', 'yellow', 'grey', 'black')]; MARKERS = 'osD^v<>phX'   # colour is never the only encoding
INK, GREY = OKABE_ITO['black'], OKABE_ITO['grey']
TWO_COLUMN = 7.35                                                          # figure width (in)


def classes_of(kinds):
    """Trap types in display order: the known types first, then the rest alphabetically."""
    kinds = set(kinds); return [k for k in KNOWN if k in kinds] + sorted(kinds - set(KNOWN))


def palette(classes):
    """Colour and marker of each trap type."""
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}, {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(classes)}


def two_line(name):
    """A trap-type name on two lines for a narrow axis."""
    fixed = {'substitutional': 'substitu-\ntional', 'interstitial': 'inter-\nstitial'}
    return fixed.get(name, name.replace(' ', '\n', 1) if ' ' in name else name)


def apply():
    """Install the rcParams."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'axes.linewidth': 0.8, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
                         'legend.frameon': True, 'legend.fancybox': False, 'legend.framealpha': 1.0, 'legend.facecolor': 'white', 'legend.edgecolor': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def style_axes(ax, minor_x=True, minor_y=True):
    """Inward ticks on all four spines; minor ticks on linear quantitative axes."""
    for side in ('top', 'right', 'bottom', 'left'): ax.spines[side].set_visible(True)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if minor_x and ax.get_xscale() == 'linear': ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if minor_y and ax.get_yscale() == 'linear': ax.yaxis.set_minor_locator(AutoMinorLocator(5))


def legend(ax, **kw):
    """Framed legend."""
    leg = ax.legend(**kw); leg.get_frame().set_linewidth(0.8); return leg


def panel_label(ax, label, x=-0.16, y=1.03):
    """Bold panel letter outside the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold', clip_on=False)


def save(fig, out, figures, name):
    """PDF and 300 dpi PNG to the results and, when figures is a folder, a 200 dpi PNG there; marked as synthetic data, without software or date stamps."""
    fig.text(0.995, 1.0, 'synthetic data', ha='right', va='bottom', fontsize=7, color=GREY); png = dict(metadata={'Software': None}, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None}, bbox_inches='tight'); fig.savefig(out / f'{name}.png', dpi=300, **png)
    if figures: fig.savefig(figures / f'{name}.png', dpi=200, **png)
    plt.close(fig)
