import numpy as np

from nmf.initialization import generate_A

m = 1000
n = 500
k = 20
seed = 42

A = generate_A(
    m=m,
    n=n,
    k_true=k,
    seed=seed
)

print(
    f"A_sum={float(np.sum(A)):.6f}"
)

print(
    f"A_norm={float(np.linalg.norm(A, 'fro')):.6f}"
)
