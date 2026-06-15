import numpy as np
from mpi4py import MPI


def distributed_AH(Aij, Hj, grid):
    local = (Aij @ Hj).astype(np.float32)

    if grid.row_comm.Get_size() == 1:
        return local

    result = np.empty_like(local, dtype=np.float32)
    grid.row_comm.Allreduce(local, result, op=MPI.SUM)
    return result


def distributed_ATW(Aij, Wi, grid):
    local = (Aij.T @ Wi).astype(np.float32)

    if grid.col_comm.Get_size() == 1:
        return local

    result = np.empty_like(local, dtype=np.float32)
    grid.col_comm.Allreduce(local, result, op=MPI.SUM)
    return result


def distributed_HtH(Hj, grid):
    k = Hj.shape[1]

    if grid.i == 0:
        local = (Hj.T @ Hj).astype(np.float32)
    else:
        local = np.zeros((k, k), dtype=np.float32)

    if grid.comm.Get_size() == 1:
        return local

    result = np.empty((k, k), dtype=np.float32)
    grid.comm.Allreduce(local, result, op=MPI.SUM)
    return result


def distributed_WtW(Wi, grid):
    k = Wi.shape[1]

    if grid.j == 0:
        local = (Wi.T @ Wi).astype(np.float32)
    else:
        local = np.zeros((k, k), dtype=np.float32)

    if grid.comm.Get_size() == 1:
        return local

    result = np.empty((k, k), dtype=np.float32)
    grid.comm.Allreduce(local, result, op=MPI.SUM)
    return result


def distributed_fro_norm(X_local, grid):
    local_sq = np.array(np.sum(X_local * X_local), dtype=np.float64)

    if grid.comm.Get_size() == 1:
        return float(np.sqrt(local_sq))

    global_sq = np.array(0.0, dtype=np.float64)
    grid.comm.Allreduce(local_sq, global_sq, op=MPI.SUM)
    return float(np.sqrt(global_sq))


def distributed_error(Aij, Wi, Hj, grid):
    residual = Aij - Wi @ Hj.T
    local_sq = np.array(np.sum(residual * residual), dtype=np.float64)

    if grid.comm.Get_size() == 1:
        return float(np.sqrt(local_sq))

    global_sq = np.array(0.0, dtype=np.float64)
    grid.comm.Allreduce(local_sq, global_sq, op=MPI.SUM)
    return float(np.sqrt(global_sq))


def distributed_factor_norms(Wi, Hj, grid):
    local_w_sq = np.array(
        np.sum(Wi * Wi) if grid.j == 0 else 0.0,
        dtype=np.float64
    )
    local_h_sq = np.array(
        np.sum(Hj * Hj) if grid.i == 0 else 0.0,
        dtype=np.float64
    )

    global_w_sq = np.array(0.0, dtype=np.float64)
    global_h_sq = np.array(0.0, dtype=np.float64)

    grid.comm.Allreduce(local_w_sq, global_w_sq, op=MPI.SUM)
    grid.comm.Allreduce(local_h_sq, global_h_sq, op=MPI.SUM)

    return (
        float(np.sqrt(global_w_sq)),
        float(np.sqrt(global_h_sq))
    )
