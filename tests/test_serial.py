import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from nmf.initialization import generate_A as generate_A_package
from nmf.solvers import solve_regularized_normal_eq
from nmf_serial import generate_A as generate_A_serial


def check(name, ok, detail):
    status = "✅" if ok else "❌"
    print(f"{status} {name:<28} {detail}", flush=True)
    assert ok, detail


def main():
    print("🧪 SERIAL CHECKS")
    print("=" * 72)

    m = 1000
    n = 500
    k = 20
    seed = 42

    A_pkg = generate_A_package(m, n, k, seed)
    A_serial = generate_A_serial(m, n, k, seed)
    diff = A_pkg - A_serial

    check(
        "A package vs serial",
        np.array_equal(A_pkg, A_serial),
        f"max_abs_diff={float(np.max(np.abs(diff))):.3e}"
    )

    gram = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float32
    )
    rhs = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32
    )
    x = solve_regularized_normal_eq(gram, rhs)

    check(
        "regularized solve shape",
        x.shape == rhs.shape and x.dtype == np.float32,
        f"shape={x.shape} dtype={x.dtype}"
    )

    print("-" * 72)
    print(f"📌 A_norm={np.linalg.norm(A_pkg):.6f} A_sum={float(np.sum(A_pkg)):.6f}")
    print("🎉 Serial checks completed")


if __name__ == "__main__":
    main()
