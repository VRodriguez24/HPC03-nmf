from mpi4py import MPI

from nmf.partition import (
    create_grid,
    local_range
)

comm = MPI.COMM_WORLD

grid = create_grid(
    comm,
    pr=2,
    pc=2
)

row_start, row_end, local_rows = local_range(
    1000,
    grid.pr,
    grid.i
)

col_start, col_end, local_cols = local_range(
    500,
    grid.pc,
    grid.j
)

print(
    f"rank={grid.rank} "
    f"coords=({grid.i},{grid.j}) "
    f"rows=[{row_start},{row_end}) "
    f"cols=[{col_start},{col_end}) "
    f"shape=({local_rows},{local_cols})",
    flush=True
)