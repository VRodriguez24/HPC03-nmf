import numpy as np
from mpi4py import MPI

from nmf.solvers import solve_regularized_normal_eq
from nmf.profiling import empty_profile
from nmf.communication import (
    distributed_AH,
    distributed_ATW,
    distributed_HtH,
    distributed_WtW,
    distributed_fro_norm,
    distributed_error
)


def nmf_als_distributed(
    Aij,
    Wi,
    Hj,
    grid,
    tol=1e-4,
    max_iter=1000,
    verbose=False,
    collect_profile=False
):

    k = Wi.shape[1]

    norm_A = distributed_fro_norm(
        Aij,
        grid
    )

    prev_err = None
    n_iter = max_iter
    profile = empty_profile()

    for it in range(max_iter):
        t_iter = MPI.Wtime()

        # ==================================================
        # UPDATE W
        # ==================================================

        t0 = MPI.Wtime()
        GH = distributed_HtH(
            Hj,
            grid
        )
        profile["hth"] += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        W_tilde = distributed_AH(
            Aij,
            Hj,
            grid
        )
        profile["ah"] += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        X = solve_regularized_normal_eq(
            GH,
            W_tilde.T
        )
        profile["solve_w"] += MPI.Wtime() - t0

        Wi = np.maximum(
            0.0,
            X.T
        )

        # ==================================================
        # UPDATE H
        # ==================================================

        t0 = MPI.Wtime()
        GW = distributed_WtW(
            Wi,
            grid
        )
        profile["wtw"] += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        H_tilde = distributed_ATW(
            Aij,
            Wi,
            grid
        )
        profile["atw"] += MPI.Wtime() - t0

        t0 = MPI.Wtime()
        X = solve_regularized_normal_eq(
            GW,
            H_tilde.T
        )
        profile["solve_h"] += MPI.Wtime() - t0

        Hj = np.maximum(
            0.0,
            X.T
        )

        # ==================================================
        # ERROR
        # ==================================================

        t0 = MPI.Wtime()
        err = distributed_error(
            Aij,
            Wi,
            Hj,
            grid
        )
        profile["error"] += MPI.Wtime() - t0

        if prev_err is None:

            delta = float("inf")

        else:

            delta = abs(
                err - prev_err
            ) / prev_err

        prev_err = err

        iter_time = MPI.Wtime() - t_iter

        if verbose and grid.is_global_root:

            print(
                f"iter={it + 1:3d} "
                f"error_rel={err / norm_A:.6f} "
                f"delta={delta:.2e} "
                f"time={iter_time:.4f}s",
                flush=True
            )

        if delta < tol:

            n_iter = it + 1

            if verbose and grid.is_global_root:

                print(
                    f"[distributed] convergencia en iteracion "
                    f"{n_iter} (delta={delta:.2e})",
                    flush=True
                )

            break

    result = (
        Wi,
        Hj,
        n_iter,
        err,
        norm_A
    )

    if collect_profile:
        return result + (profile,)

    return result
