import numpy as np
from mpi4py import MPI


PROFILE_KEYS = (
    "hth",
    "ah",
    "solve_w",
    "wtw",
    "atw",
    "solve_h",
    "error",
)


def empty_profile():
    return {key: 0.0 for key in PROFILE_KEYS}


def summarize_profile(profile, grid):
    summary = {}

    for key in PROFILE_KEYS:
        local = np.array(profile.get(key, 0.0), dtype=np.float64)
        global_max = np.array(0.0, dtype=np.float64)
        grid.comm.Allreduce(local, global_max, op=MPI.MAX)
        summary[key] = float(global_max)

    summary["total_profiled"] = sum(summary.values())
    return summary


def print_profile(profile, grid):
    if not grid.is_global_root:
        return

    total = max(profile.get("total_profiled", 0.0), 1e-12)

    print()
    print("⏱️  PROFILE")
    print("-" * 60)

    for key in PROFILE_KEYS:
        value = profile.get(key, 0.0)
        pct = 100.0 * value / total
        print(f"{key:>10s} : {value:10.6f}s  {pct:6.2f}%")

    print("-" * 60)
