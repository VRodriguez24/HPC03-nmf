import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import generate_A_block
from nmf.communication import distributed_fro_norm


comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

Aij = generate_A_block(
    m=1000,
    n=500,
    k_true=20,
    seed=42,
    grid=grid
)

norm_A = distributed_fro_norm(
    Aij,
    grid
)

print(
    f"rank={grid.rank} "
    f"norm_A={norm_A:.6f}",
    flush=True
)