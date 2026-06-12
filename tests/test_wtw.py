import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import initialize_W_block
from nmf.communication import distributed_WtW


comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

m = 1000
k = 20

Wi = initialize_W_block(
    m=m,
    k=k,
    grid=grid
)

GW = distributed_WtW(
    Wi=Wi,
    grid=grid
)

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"GW={GW.shape} "
    f"sum={float(np.sum(GW)):.6f} "
    f"norm={float(np.linalg.norm(GW)):.6f}",
    flush=True
)