from dataclasses import dataclass


@dataclass
class Grid:
    """
    Representa la grilla lógica de procesos MPI de tamaño pr x pc.
    """

    rank: int
    size: int

    pr: int
    pc: int

    i: int
    j: int

    row_comm: object
    col_comm: object

    @property
    def is_row_root(self):
        """Proceso raíz dentro de cada fila."""
        return self.j == 0

    @property
    def is_col_root(self):
        """Proceso raíz dentro de cada columna."""
        return self.i == 0

    @property
    def is_global_root(self):
        """Proceso raíz global."""
        return self.rank == 0


def local_range(global_size, num_parts, part_id):
    """
    Divide un rango [0, global_size) en num_parts bloques iguales.
    Retorna:
        start, end, local_size
    """

    local_size = global_size // num_parts

    start = part_id * local_size
    end = start + local_size

    return start, end, local_size


def create_grid(comm, pr, pc):
    """
    Construye la grilla lógica MPI y los comunicadores
    por filas y columnas.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != pr * pc:
        raise ValueError(
            f"size={size} pero pr*pc={pr * pc}"
        )

    i = rank // pc
    j = rank % pc

    row_comm = comm.Split(
        color=i,
        key=j
    )

    col_comm = comm.Split(
        color=j,
        key=i
    )

    return Grid(
        rank=rank,
        size=size,

        pr=pr,
        pc=pc,

        i=i,
        j=j,

        row_comm=row_comm,
        col_comm=col_comm
    )
