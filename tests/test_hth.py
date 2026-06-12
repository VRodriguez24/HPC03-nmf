import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import initialize_H_block
from nmf.communication import distributed_HtH


comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

n = 500
k = 20

Hj = initialize_H_block(
    n=n,
    k=k,
    grid=grid
)

GH = distributed_HtH(
    Hj=Hj,
    grid=grid
)

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"GH={GH.shape} "
    f"sum={float(np.sum(GH)):.6f} "
    f"norm={float(np.linalg.norm(GH)):.6f}",
    flush=True
)