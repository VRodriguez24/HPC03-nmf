import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import (
    generate_A_block,
    initialize_H_block
)
from nmf.communication import distributed_AH


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

Hj = initialize_H_block(
    n=n,
    k=k,
    grid=grid
)

W_tilde_i = distributed_AH(
    Aij=Aij,
    Hj=Hj,
    grid=grid
)

local_sum = np.array(
    np.sum(W_tilde_i),
    dtype=np.float64
)

local_norm_sq = np.array(
    np.sum(W_tilde_i * W_tilde_i),
    dtype=np.float64
)

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"W_tilde={W_tilde_i.shape} "
    f"sum={float(local_sum):.6f} "
    f"norm={float(np.sqrt(local_norm_sq)):.6f}",
    flush=True
)