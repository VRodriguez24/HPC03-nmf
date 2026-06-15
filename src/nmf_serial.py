import numpy as np
import argparse
import time

from nmf.solvers import solve_regularized_normal_eq


def generate_A(m, n, k_true=20, seed=42):
    rng_W = np.random.default_rng(seed)
    W_true = rng_W.random((m, k_true)).astype(np.float32)

    rng_H = np.random.default_rng(seed + 1)
    H_true = rng_H.random((n, k_true)).astype(np.float32)

    A = (W_true @ H_true.T).astype(np.float32)

    noise = np.empty((m, n), dtype=np.float32)
    for r in range(m):
        rng_noise = np.random.default_rng(seed * 1000 + r)
        noise[r] = rng_noise.random(n)
    A += (0.01 * noise).astype(np.float32)

    return np.maximum(0.0, A)


def nmf_als(A, k, tol=1e-4, max_iter=1000, verbose=False):
    m, n = A.shape

    rng_W = np.random.default_rng(99)
    W = rng_W.random((m, k)).astype(np.float32)

    rng_H = np.random.default_rng(999)
    H = rng_H.random((n, k)).astype(np.float32)

    print(f"[serial] check W0: norm_F={np.linalg.norm(W, 'fro'):.6f}  sum={float(np.sum(W)):.6f}", flush=True)
    print(f"[serial] check H0: norm_F={np.linalg.norm(H, 'fro'):.6f}  sum={float(np.sum(H)):.6f}", flush=True)

    if verbose:
        norm_A = np.linalg.norm(A, 'fro')

    prev_err = None
    n_iter   = max_iter

    for it in range(max_iter):
        GH      = (H.T @ H).astype(np.float32)
        W_tilde = (A @ H).astype(np.float32)

        if verbose and it == 0:
            print(f"[debug serial] GH norm={np.linalg.norm(GH):.6f} sum={float(np.sum(GH)):.6f}")
            print(f"[debug serial] W_tilde norm={np.linalg.norm(W_tilde):.6f} sum={float(np.sum(W_tilde)):.6f}")

        X       = solve_regularized_normal_eq(GH, W_tilde.T)
        W       = np.maximum(0.0, X.T)

        if verbose and it == 0:
            print(f"[debug serial] W norm={np.linalg.norm(W):.6f} sum={float(np.sum(W)):.6f}")

        GW      = (W.T @ W).astype(np.float32)
        H_tilde = (A.T @ W).astype(np.float32)

        if verbose and it == 0:
            print(f"[debug serial] GW norm={np.linalg.norm(GW):.6f} sum={float(np.sum(GW)):.6f}")
            print(f"[debug serial] H_tilde norm={np.linalg.norm(H_tilde):.6f} sum={float(np.sum(H_tilde)):.6f}")

        X       = solve_regularized_normal_eq(GW, H_tilde.T)
        H       = np.maximum(0.0, X.T)

        if verbose and it == 0:
            print(f"[debug serial] H norm={np.linalg.norm(H):.6f} sum={float(np.sum(H)):.6f}")

        err   = np.linalg.norm(A - W @ H.T, 'fro')
        delta = abs(err - prev_err) / prev_err if prev_err is not None else float('inf')
        prev_err = err

        if verbose:
            print(f"  iter {it+1:3d}  error_rel={err/norm_A:.6f}  delta={delta:.2e}", flush=True)

        if delta < tol:
            n_iter = it + 1
            if verbose:
                print(f"[serial] convergencia en iteracion {n_iter} (delta={delta:.2e})", flush=True)
            break

    return W, H, n_iter, err


def main():
    parser = argparse.ArgumentParser(description="NMF serial")
    parser.add_argument("--m",        type=int,   default=20000)
    parser.add_argument("--n",        type=int,   default=5000)
    parser.add_argument("--k",        type=int,   default=20)
    parser.add_argument("--tol",      type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int,   default=1000)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print(f"[serial] generando A ({args.m}x{args.n})...", flush=True)
    A = generate_A(args.m, args.n, k_true=args.k, seed=args.seed)

    norm_A = np.linalg.norm(A, 'fro')
    print(f"[serial] check A: norm_F={norm_A:.6f}  sum={float(np.sum(A)):.6f}", flush=True)
    print(f"[serial] iniciando ALS (max_iter={args.max_iter}, tol={args.tol})", flush=True)

    t0 = time.perf_counter()
    W, H, n_iter, err = nmf_als(A, args.k, tol=args.tol, max_iter=args.max_iter, verbose=args.verbose)
    t1 = time.perf_counter()

    print(f"[serial] check W: norm_F={np.linalg.norm(W, 'fro'):.6f}  sum={float(np.sum(W)):.6f}", flush=True)
    print(f"[serial] check H: norm_F={np.linalg.norm(H, 'fro'):.6f}  sum={float(np.sum(H)):.6f}", flush=True)
    print(
        f"m={args.m} n={args.n} k={args.k} "
        f"iter={n_iter} "
        f"tiempo_total={t1 - t0:.3f}s "
        f"tiempo_por_iter={(t1 - t0) / n_iter:.4f}s "
        f"error_rel={err / norm_A:.6f}",
        flush=True
    )


if __name__ == "__main__":
    main()
