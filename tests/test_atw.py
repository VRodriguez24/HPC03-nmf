import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import (
    generate_A_block,
    initialize_W_block
)
from nmf.communication import distributed_ATW


comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

m = 1000
n = 500
k = 20
seed = 42

Aij = generate_A_block(
    m=m,
    n=n,
    k_true=k,
    seed=seed,
    grid=grid
)

Wi = initialize_W_block(
    m=m,
    k=k,
    grid=grid
)

H_tilde_j = distributed_ATW(
    Aij=Aij,
    Wi=Wi,
    grid=grid
)

local_sum = np.array(
    np.sum(H_tilde_j),
    dtype=np.float64
)

local_norm_sq = np.array(
    np.sum(H_tilde_j * H_tilde_j),
    dtype=np.float64
)

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"H_tilde={H_tilde_j.shape} "
    f"sum={float(local_sum):.6f} "
    f"norm={float(np.sqrt(local_norm_sq)):.6f}",
    flush=True
)