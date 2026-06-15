import argparse
import csv
import os

from mpi4py import MPI

from nmf.partition import create_grid

from nmf.initialization import (
    generate_A_block,
    initialize_W_block,
    initialize_H_block
)

from nmf.als import (
    nmf_als_distributed
)
from nmf.communication import distributed_factor_norms
from nmf.profiling import (
    print_profile,
    summarize_profile
)


def parse_args():
    """
    Procesa los argumentos de línea de comandos.
    """

    parser = argparse.ArgumentParser(
        description="Distributed NMF using MPI and ALS"
    )

    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="Número de filas de procesos"
    )

    parser.add_argument(
        "--pc",
        type=int,
        required=True,
        help="Número de columnas de procesos"
    )

    parser.add_argument(
        "--m",
        type=int,
        default=20000,
        help="Número de filas de A"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Número de columnas de A"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Rango de la factorización"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para generación de datos"
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerancia de convergencia"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Máximo número de iteraciones"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar progreso ALS"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta opcional para guardar una fila CSV de resultados"
    )

    return parser.parse_args()


def initialize_problem(args, grid):
    """
    Construye los bloques locales necesarios
    para ejecutar ALS distribuido.
    """

    Aij = generate_A_block(
        m=args.m,
        n=args.n,
        k_true=args.k,
        seed=args.seed,
        grid=grid
    )

    Wi = initialize_W_block(
        m=args.m,
        k=args.k,
        grid=grid
    )

    Hj = initialize_H_block(
        n=args.n,
        k=args.k,
        grid=grid
    )

    return Aij, Wi, Hj



def run_nmf(args, grid):
    """
    Ejecuta NMF distribuido.
    """

    Aij, Wi, Hj = initialize_problem(
        args,
        grid
    )

    grid.comm.Barrier()

    t0 = MPI.Wtime()

    Wi, Hj, n_iter, err, norm_A, profile = (
        nmf_als_distributed(
            Aij=Aij,
            Wi=Wi,
            Hj=Hj,
            grid=grid,
            tol=args.tol,
            max_iter=args.max_iter,
            verbose=args.verbose,
            collect_profile=True
        )
    )

    grid.comm.Barrier()

    elapsed = MPI.Wtime() - t0

    norm_W, norm_H = distributed_factor_norms(
        Wi,
        Hj,
        grid
    )

    profile = summarize_profile(
        profile,
        grid
    )

    return (
        Wi,
        Hj,
        n_iter,
        err,
        norm_A,
        norm_W,
        norm_H,
        profile,
        elapsed
    )


def print_summary(
    args,
    grid,
    n_iter,
    err,
    norm_A,
    norm_W,
    norm_H,
    profile,
    elapsed
):
    """
    Imprime resultados finales.
    """

    if not grid.is_global_root:
        return

    print()

    print("=" * 60)
    print("DISTRIBUTED NMF RESULTS")
    print("=" * 60)

    print(
        f"process_grid : {args.pr} x {args.pc}"
    )

    print(
        f"matrix_size  : "
        f"{args.m} x {args.n}"
    )

    print(
        f"rank_k       : {args.k}"
    )

    print(
        f"iterations   : {n_iter}"
    )

    print(
        f"error        : {err:.6f}"
    )

    print(
        f"error_rel    : "
        f"{err / norm_A:.6f}"
    )

    print(
        f"norm_A       : "
        f"{norm_A:.6f}"
    )

    print(
        f"norm_W       : "
        f"{norm_W:.6f}"
    )

    print(
        f"norm_H       : "
        f"{norm_H:.6f}"
    )

    print(
        f"time_sec     : "
        f"{elapsed:.4f}"
    )

    print(
        f"time_per_iter: "
        f"{elapsed / n_iter:.6f}"
    )

    print("=" * 60)

    print_profile(
        profile,
        grid
    )


def write_csv(args, grid, n_iter, err, norm_A, norm_W, norm_H, profile, elapsed):
    if args.csv is None or not grid.is_global_root:
        return

    fieldnames = [
        "pr",
        "pc",
        "threads",
        "m",
        "n",
        "k",
        "seed",
        "iterations",
        "elapsed_sec",
        "time_per_iter",
        "error",
        "error_rel",
        "norm_A",
        "norm_W",
        "norm_H",
        "profile_hth",
        "profile_ah",
        "profile_solve_w",
        "profile_wtw",
        "profile_atw",
        "profile_solve_h",
        "profile_error",
    ]

    row = {
        "pr": args.pr,
        "pc": args.pc,
        "threads": os.environ.get("OMP_NUM_THREADS", ""),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "seed": args.seed,
        "iterations": n_iter,
        "elapsed_sec": elapsed,
        "time_per_iter": elapsed / n_iter,
        "error": err,
        "error_rel": err / norm_A,
        "norm_A": norm_A,
        "norm_W": norm_W,
        "norm_H": norm_H,
        "profile_hth": profile["hth"],
        "profile_ah": profile["ah"],
        "profile_solve_w": profile["solve_w"],
        "profile_wtw": profile["wtw"],
        "profile_atw": profile["atw"],
        "profile_solve_h": profile["solve_h"],
        "profile_error": profile["error"],
    }

    file_exists = os.path.exists(args.csv)

    with open(args.csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():

    comm = MPI.COMM_WORLD

    args = parse_args()

    grid = create_grid(
        comm=comm,
        pr=args.pr,
        pc=args.pc
    )

    (
        Wi,
        Hj,
        n_iter,
        err,
        norm_A,
        norm_W,
        norm_H,
        profile,
        elapsed
    ) = run_nmf(
        args,
        grid
    )

    print_summary(
        args,
        grid,
        n_iter,
        err,
        norm_A,
        norm_W,
        norm_H,
        profile,
        elapsed
    )

    write_csv(
        args,
        grid,
        n_iter,
        err,
        norm_A,
        norm_W,
        norm_H,
        profile,
        elapsed
    )


if __name__ == "__main__":
    main()
