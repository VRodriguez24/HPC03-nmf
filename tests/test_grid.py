from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pr", type=int, required=True)
parser.add_argument("--pc", type=int, required=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

assert size == args.pr * args.pc

i = rank // args.pc
j = rank % args.pc

row_comm = comm.Split(i, j)
col_comm = comm.Split(j, i)

print(
    f"global_rank={rank} "
    f"coords=({i},{j}) "
    f"row_rank={row_comm.Get_rank()} "
    f"col_rank={col_comm.Get_rank()}",
    flush=True
)