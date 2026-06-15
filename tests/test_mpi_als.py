import argparse
import os
import sys

from mpi4py import MPI


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from nmf.als import nmf_als_distributed
from nmf.communication import distributed_factor_norms
from nmf.initialization import (
    generate_A_block,
    initialize_H_block,
    initialize_W_block,
)
from nmf.partition import create_grid
from nmf.profiling import print_profile, summarize_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=int, default=2)
    parser.add_argument("--pc", type=int, default=2)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    grid = create_grid(comm, args.pr, args.pc)

    if grid.is_global_root:
        print("🧪 MPI ALS CHECK")
        print("=" * 72)
        print(
            f"🌐 grid={args.pr}x{args.pc} ranks={grid.size} "
            f"m={args.m} n={args.n} k={args.k} max_iter={args.max_iter}",
            flush=True
        )

    Aij = generate_A_block(args.m, args.n, args.k, args.seed, grid)
    Wi = initialize_W_block(args.m, args.k, grid)
    Hj = initialize_H_block(args.n, args.k, grid)

    Wi, Hj, n_iter, err, norm_A, profile = nmf_als_distributed(
        Aij=Aij,
        Wi=Wi,
        Hj=Hj,
        grid=grid,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
        collect_profile=True,
    )

    norm_W, norm_H = distributed_factor_norms(Wi, Hj, grid)
    profile = summarize_profile(profile, grid)

    if grid.is_global_root:
        print("-" * 72)
        print(f"✅ iterations={n_iter}")
        print(f"✅ error_rel={err / norm_A:.6f}")
        print(f"✅ norm_A={norm_A:.6f} norm_W={norm_W:.6f} norm_H={norm_H:.6f}")

    print_profile(profile, grid)

    if grid.is_global_root:
        print("🎉 MPI ALS check completed")


if __name__ == "__main__":
    main()
