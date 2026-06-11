import numpy as np

from nmf.partition import local_range


def initialize_W_block(m, k, grid):
    """
    Inicializa el bloque local Wi.

    Reproduce exactamente la inicialización del código serial:
        rng_W = np.random.default_rng(99)
        W = rng_W.random((m, k))

    Cada fila lógica de procesos comparte el mismo Wi.
    """

    row_start, row_end, _ = local_range(
        m,
        grid.pr,
        grid.i
    )

    rng_W = np.random.default_rng(99)

    W_full = rng_W.random(
        (m, k)
    ).astype(np.float32)

    return W_full[row_start:row_end].copy()


def initialize_H_block(n, k, grid):
    """
    Inicializa el bloque local Hj.

    Reproduce exactamente la inicialización del código serial:
        rng_H = np.random.default_rng(999)
        H = rng_H.random((n, k))

    Cada columna lógica de procesos comparte el mismo Hj.
    """

    col_start, col_end, _ = local_range(
        n,
        grid.pc,
        grid.j
    )

    rng_H = np.random.default_rng(999)

    H_full = rng_H.random(
        (n, k)
    ).astype(np.float32)

    return H_full[col_start:col_end].copy()


def generate_A(m, n, k_true, seed):
    """
    Genera la matriz serial A completa.

    Usa la misma lógica determinística que generate_A_block:

        A = W_true @ H_true.T + 0.01 * noise
    """

    rng_W = np.random.default_rng(seed)
    W_true = rng_W.random(
        (m, k_true)
    ).astype(np.float32)

    rng_H = np.random.default_rng(seed + 1)
    H_true = rng_H.random(
        (n, k_true)
    ).astype(np.float32)

    A = (
        W_true @ H_true.T
    ).astype(np.float32)

    noise = np.empty(
        (m, n),
        dtype=np.float32
    )

    for row in range(m):
        rng_noise = np.random.default_rng(
            seed * 1000 + row
        )

        noise[row, :] = rng_noise.random(
            n
        ).astype(np.float32)

    A += (
        0.01 * noise
    ).astype(np.float32)

    return np.maximum(
        0.0,
        A
    )


def generate_A_block(m, n, k_true, seed, grid):
    """
    Genera el bloque local Aij de la matriz A.

    Replica la lógica del código serial:

        A = W_true @ H_true.T + 0.01 * noise

    pero construyendo solo el bloque correspondiente al proceso (i,j).
    """

    row_start, row_end, local_m = local_range(
        m,
        grid.pr,
        grid.i
    )

    col_start, col_end, local_n = local_range(
        n,
        grid.pc,
        grid.j
    )

    rng_W = np.random.default_rng(seed)
    W_true_full = rng_W.random(
        (m, k_true)
    ).astype(np.float32)

    W_true_local = W_true_full[
        row_start:row_end,
        :
    ]

    rng_H = np.random.default_rng(seed + 1)
    H_true_full = rng_H.random(
        (n, k_true)
    ).astype(np.float32)

    H_true_local = H_true_full[
        col_start:col_end,
        :
    ]

    Aij = (
        W_true_local @ H_true_local.T
    ).astype(np.float32)

    noise = np.empty(
        (local_m, local_n),
        dtype=np.float32
    )

    for local_r, global_r in enumerate(
        range(row_start, row_end)
    ):
        rng_noise = np.random.default_rng(
            seed * 1000 + global_r
        )

        noise_row_full = rng_noise.random(
            n
        ).astype(np.float32)

        noise[local_r, :] = noise_row_full[
            col_start:col_end
        ]

    Aij += (
        0.01 * noise
    ).astype(np.float32)

    return np.maximum(
        0.0,
        Aij
    )
