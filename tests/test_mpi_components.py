import argparse
import os
import sys

import numpy as np
from mpi4py import MPI


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from nmf.communication import (
    distributed_AH,
    distributed_ATW,
    distributed_HtH,
    distributed_WtW,
    distributed_error,
    distributed_factor_norms,
    distributed_fro_norm,
)
from nmf.initialization import (
    generate_A,
    generate_A_block,
    initialize_H_block,
    initialize_W_block,
)
from nmf.partition import create_grid, local_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=int, default=2)
    parser.add_argument("--pc", type=int, default=2)
    parser.add_argument("--m", type=int, default=80)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def root_check(grid, name, ok, detail):
    if grid.is_global_root:
        status = "✅" if ok else "❌"
        print(f"{status} {name:<28} {detail}", flush=True)
    assert ok, detail


def all_ranks_ok(local_ok, grid):
    local = np.array(1 if local_ok else 0, dtype=np.int32)
    total = np.array(0, dtype=np.int32)
    grid.comm.Allreduce(local, total, op=MPI.SUM)
    return int(total) == grid.size


def global_max(value, grid):
    local = np.array(value, dtype=np.float64)
    result = np.array(0.0, dtype=np.float64)
    grid.comm.Allreduce(local, result, op=MPI.MAX)
    return float(result)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    grid = create_grid(comm, args.pr, args.pc)

    if grid.is_global_root:
        print("🧪 MPI COMPONENT CHECKS")
        print("=" * 72)
        print(
            f"🌐 grid={args.pr}x{args.pc} ranks={grid.size} "
            f"m={args.m} n={args.n} k={args.k}",
            flush=True
        )

    Aij = generate_A_block(args.m, args.n, args.k, args.seed, grid)
    Wi = initialize_W_block(args.m, args.k, grid)
    Hj = initialize_H_block(args.n, args.k, grid)

    row_start, row_end, _ = local_range(args.m, args.pr, grid.i)
    col_start, col_end, _ = local_range(args.n, args.pc, grid.j)

    A = generate_A(args.m, args.n, args.k, args.seed)
    rng_w = np.random.default_rng(99)
    W = rng_w.random((args.m, args.k)).astype(np.float32)
    rng_h = np.random.default_rng(999)
    H = rng_h.random((args.n, args.k)).astype(np.float32)

    root_check(
        grid,
        "local shapes",
        all_ranks_ok(
            Aij.shape == (args.m // args.pr, args.n // args.pc)
            and Wi.shape == (args.m // args.pr, args.k)
            and Hj.shape == (args.n // args.pc, args.k),
            grid
        ),
        f"Aij={Aij.shape} Wi={Wi.shape} Hj={Hj.shape}"
    )

    A_slice = A[row_start:row_end, col_start:col_end]
    A_diff = np.max(np.abs(Aij - A_slice))
    A_block_ok = np.allclose(Aij, A_slice, rtol=1e-6, atol=1e-5)
    A_block_max_diff = global_max(A_diff, grid)

    root_check(
        grid,
        "A block",
        all_ranks_ok(A_block_ok, grid),
        f"max_abs_diff={A_block_max_diff:.3e}"
    )

    GH = distributed_HtH(Hj, grid)
    GW = distributed_WtW(Wi, grid)
    AH = distributed_AH(Aij, Hj, grid)
    ATW = distributed_ATW(Aij, Wi, grid)

    GH_diff = global_max(np.max(np.abs(GH - H.T @ H)), grid)
    root_check(
        grid,
        "HᵀH",
        all_ranks_ok(np.allclose(GH, H.T @ H, rtol=1e-5, atol=1e-3), grid),
        f"norm={np.linalg.norm(GH):.6f} max_abs_diff={GH_diff:.3e}"
    )

    GW_diff = global_max(np.max(np.abs(GW - W.T @ W)), grid)
    root_check(
        grid,
        "WᵀW",
        all_ranks_ok(np.allclose(GW, W.T @ W, rtol=1e-5, atol=1e-3), grid),
        f"norm={np.linalg.norm(GW):.6f} max_abs_diff={GW_diff:.3e}"
    )

    AH_ref = (A @ H)[row_start:row_end, :]
    AH_diff = global_max(np.max(np.abs(AH - AH_ref)), grid)
    root_check(
        grid,
        "AH",
        all_ranks_ok(
            np.allclose(AH, AH_ref, rtol=1e-5, atol=1e-2),
            grid
        ),
        f"local_shape={AH.shape} max_abs_diff={AH_diff:.3e}"
    )

    ATW_ref = (A.T @ W)[col_start:col_end, :]
    ATW_diff = global_max(np.max(np.abs(ATW - ATW_ref)), grid)
    root_check(
        grid,
        "AᵀW",
        all_ranks_ok(
            np.allclose(ATW, ATW_ref, rtol=1e-5, atol=1e-2),
            grid
        ),
        f"local_shape={ATW.shape} max_abs_diff={ATW_diff:.3e}"
    )

    norm_A = distributed_fro_norm(Aij, grid)
    norm_W, norm_H = distributed_factor_norms(Wi, Hj, grid)
    err = distributed_error(Aij, Wi, Hj, grid)

    if grid.is_global_root:
        print("-" * 72)
        print(f"📌 norm_A={norm_A:.6f} norm_W={norm_W:.6f} norm_H={norm_H:.6f}")
        print(f"📌 initial_error={err:.6f}")
        print("🎉 MPI component checks completed")


if __name__ == "__main__":
    main()
