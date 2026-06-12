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

def distributed_HtH(Hj, grid):
    """
    Calcula H.T @ H de forma distribuida.

    Como cada bloque Hj está replicado en todas las filas de la grilla,
    solo los procesos de la primera fila (i == 0) contribuyen para
    evitar duplicar términos.

    El resultado queda disponible en todos los procesos.
    """

    k = Hj.shape[1]

    if grid.i == 0:
        local = (
            Hj.T @ Hj
        ).astype(np.float32)
    else:
        local = np.zeros(
            (k, k),
            dtype=np.float32
        )

    result = np.empty(
        (k, k),
        dtype=np.float32
    )

    grid.row_comm.Get_parent()

    grid.row_comm.Barrier()

    # Se usa el comunicador global implícito a partir de row/col no existe aquí,
    # por eso necesitamos que Grid tenga también comm global.
    
def distributed_HtH(Hj, grid):
    """
    Calcula H.T @ H de forma distribuida.

    Como cada bloque Hj está replicado en todas las filas de la grilla,
    solo los procesos de la primera fila contribuyen para evitar duplicación.

    El resultado queda disponible en todos los procesos.
    """

    k = Hj.shape[1]

    if grid.i == 0:
        local = (
            Hj.T @ Hj
        ).astype(np.float32)
    else:
        local = np.zeros(
            (k, k),
            dtype=np.float32
        )

    result = np.empty(
        (k, k),
        dtype=np.float32
    )

    grid.comm.Allreduce(
        local,
        result,
        op=MPI.SUM
    )

    return result


def distributed_WtW(Wi, grid):
    """
    Calcula W.T @ W de forma distribuida.

    Como cada bloque Wi está replicado en todas las columnas de la grilla,
    solo los procesos de la primera columna contribuyen para evitar duplicación.

    El resultado queda disponible en todos los procesos.
    """

    k = Wi.shape[1]

    if grid.j == 0:
        local = (
            Wi.T @ Wi
        ).astype(np.float32)
    else:
        local = np.zeros(
            (k, k),
            dtype=np.float32
        )

    result = np.empty(
        (k, k),
        dtype=np.float32
    )

    grid.comm.Allreduce(
        local,
        result,
        op=MPI.SUM
    )

    return result

def distributed_fro_norm(X_local, grid):
    """
    Calcula la norma de Frobenius global de una matriz distribuida.

    Cada proceso aporta la suma de cuadrados de su bloque local.
    """

    local_sq = np.array(
        np.sum(X_local * X_local),
        dtype=np.float64
    )

    global_sq = np.array(
        0.0,
        dtype=np.float64
    )

    grid.comm.Allreduce(
        local_sq,
        global_sq,
        op=MPI.SUM
    )

    return float(
        np.sqrt(global_sq)
    )
    
def distributed_error(Aij, Wi, Hj, grid):
    """
    Calcula

        ||A - WHᵀ||_F

    de forma distribuida.
    """

    residual = (
        Aij - Wi @ Hj.T
    )

    local_sq = np.array(
        np.sum(residual * residual),
        dtype=np.float64
    )

    global_sq = np.array(
        0.0,
        dtype=np.float64
    )

    grid.comm.Allreduce(
        local_sq,
        global_sq,
        op=MPI.SUM
    )

    return float(
        np.sqrt(global_sq)
    )