from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import (
    generate_A_block,
    initialize_W_block,
    initialize_H_block
)
from nmf.communication import (
    distributed_error
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

Aij = generate_A_block(
    m=m,
    n=n,
    k_true=k,
    seed=42,
    grid=grid
)

Wi = initialize_W_block(
    m=m,
    k=k,
    grid=grid
)

Hj = initialize_H_block(
    n=n,
    k=k,
    grid=grid
)

err = distributed_error(
    Aij,
    Wi,
    Hj,
    grid
)

print(
    f"rank={grid.rank} "
    f"error={err:.6f}",
    flush=True
)