import numpy as np


REGULARIZATION = 1e-6


def solve_regularized_normal_eq(gram, rhs, reg=REGULARIZATION):
    k = gram.shape[0]
    gram64 = gram.astype(np.float64, copy=False)
    rhs64 = rhs.astype(np.float64, copy=False)

    scale = float(np.trace(gram64)) / k
    ridge = reg * max(scale, 1.0)

    lhs = gram64 + ridge * np.eye(
        k,
        dtype=np.float64
    )

    return np.linalg.solve(
        lhs,
        rhs64
    ).astype(np.float32)
