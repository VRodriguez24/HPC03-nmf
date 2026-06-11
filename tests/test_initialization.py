from mpi4py import MPI
import numpy as np

from nmf.partition import create_grid
from nmf.initialization import (
    initialize_W_block,
    initialize_H_block
)

comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

m = 1000
n = 500
k = 20

W = initialize_W_block(
    m,
    k,
    grid
)

H = initialize_H_block(
    n,
    k,
    grid
)

print(
    f"rank={grid.rank} "
    f"W={W.shape} "
    f"H={H.shape} "
    f"normW={np.linalg.norm(W):.6f} "
    f"normH={np.linalg.norm(H):.6f}",
    flush=True
)
