# sa-gw: state-aligned graph wavelets

Graph-wavelet fingerprints of charge-trapping events in amorphous networks.  For each event, a cell before and after it traps an
electron, sa-gw builds the A-B bond graph, computes wavelet scattering coefficients of structural, potential and spin channels, and
compares them at the trap atom.

    pip install -r requirements.txt
    pytest -q
    bash demo/run.sh            # example -> fingerprints -> analysis -> demo/out/ (half a minute)
    bash demo/run.sh figures    # also refreshes demo/figures/
    bash demo/run.sh remake     # regenerates the example data first (twelve minutes)

Set `OMP_NUM_THREADS=1`, as the scripts and tests do: it is much faster for many small matrices.

![workflow](demo/figures/fig00_workflow.png)

The example data are a synthetic toy (`example/data/README.md`).  To analyse your own calculations: [docs/tutorial.md](docs/tutorial.md).

    sagw/network.py        settings
    sagw/structures.py     structure files, periodic geometry, displacements
    sagw/graph.py          bond graph, diffusion operator, wavelets, scattering, structural channels, T0, eta, eta_t
    sagw/potential.py      potential cube -> site and shell potential of every atom
    sagw/spin.py           spin cube or Mulliken populations -> share of the added electron on every atom
    sagw/descriptors.py    descriptors of the structural response
    sagw/models.py         held-out regression
    sagw/sensitivity.py    sensitivity to the cutoff and the edge weight
    docs/                  tutorial
    example/               make_example.py and the synthetic data/
    demo/                  the example analysis and its figures
    tests/                 pytest; reference/ holds the demo's numbers

Licence: MIT (Olsson Materials Modelling).

## Citing sa_gw
If you use sa_gw in your work, please cite: 

"State aligned graph wavelet fingerprints of electron trapping in amorphous Si3N4", arXiv preprint arXiv:2610.06173, doi: [https://doi.org/10.48550/arXiv.2610.06173]
