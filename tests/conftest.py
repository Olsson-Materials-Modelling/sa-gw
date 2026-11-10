"""Single-threaded linear algebra for the tests unless the environment sets it: thread pools on a shared machine slow small eigendecompositions by orders of magnitude."""
import os

for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(var, '1')
