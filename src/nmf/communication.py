import numpy as np

from mpi4py import MPI


def distributed_AH(Aij, Hj, grid):
    """
    Calcula de forma distribuida el bloque local de A @ H.

    Cada proceso (i,j) calcula:

        Aij @ Hj

    Luego se suman las contribuciones dentro de cada fila de procesos:

        W_tilde_i = sum_j Aij @ Hj

    El resultado queda replicado en todos los procesos de la misma fila.
    """

    local = (
        Aij @ Hj
    ).astype(np.float32)

    result = np.empty_like(
        local,
        dtype=np.float32
    )

    grid.row_comm.Allreduce(
        local,
        result,
        op=MPI.SUM
    )

    return result


def distributed_ATW(Aij, Wi, grid):
    """
    Calcula de forma distribuida el bloque local de A.T @ W.

    Cada proceso (i,j) calcula:

        Aij.T @ Wi

    Luego se suman las contribuciones dentro de cada columna de procesos:

        H_tilde_j = sum_i Aij.T @ Wi

    El resultado queda replicado en todos los procesos de la misma columna.
    """

    local = (
        Aij.T @ Wi
    ).astype(np.float32)

    result = np.empty_like(
        local,
        dtype=np.float32
    )

    grid.col_comm.Allreduce(
        local,
        result,
        op=MPI.SUM
    )

    return result