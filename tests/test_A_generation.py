import numpy as np

from mpi4py import MPI

from nmf.partition import create_grid
from nmf.initialization import generate_A_block


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

local_sum = np.array(
    np.sum(Aij),
    dtype=np.float64
)

global_sum = np.array(
    0.0,
    dtype=np.float64
)

comm.Allreduce(
    local_sum,
    global_sum,
    op=MPI.SUM
)

local_norm_sq = np.array(
    np.sum(Aij * Aij),
    dtype=np.float64
)

global_norm_sq = np.array(
    0.0,
    dtype=np.float64
)

comm.Allreduce(
    local_norm_sq,
    global_norm_sq,
    op=MPI.SUM
)

if grid.is_global_root:
    print(
        f"A_global_sum={float(global_sum):.6f} "
        f"A_global_norm={float(np.sqrt(global_norm_sq)):.6f}",
        flush=True
    )

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"Aij={Aij.shape} "
    f"local_sum={float(local_sum):.6f}",
    flush=True
)