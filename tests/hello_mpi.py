from mpi4py import MPI

comm = MPI.COMM_WORLD

print(
    f"rank={comm.Get_rank()} "
    f"size={comm.Get_size()}"
)