# 🧮 NMF Distribuido con MPI

Tarea 3 - IIC3533 Computación de Alto Rendimiento.

Implementación de **Factorización de Matrices No Negativas (NMF)** mediante **ALS distribuido**, usando una grilla MPI 2D `pr x pc` y paralelismo NumPy por proceso.

## 📌 Objetivo

Aproximar una matriz no negativa:

```text
A ≈ W Hᵀ
```

con distribución 2D de `A`, `W` y `H`, midiendo rendimiento y reproducibilidad en distintas configuraciones MPI + threads.

## 🗂️ Estructura

```text
src/
  nmf_distributed.py      # Ejecutable MPI principal
  nmf_serial.py           # Referencia serial
  nmf/
    partition.py          # Grilla MPI y partición 2D
    initialization.py     # Inicialización reproducible
    communication.py      # Operaciones distribuidas
    als.py                # Loop ALS distribuido
    solvers.py            # Solve regularizado
    profiling.py          # Tiempos por operación

slurm/                    # Scripts de ejecución en cluster
tests/                    # Tests seriales y MPI
results/                  # CSVs por job y tabla final
outputs/                  # Salidas SLURM
```

## ✅ Tests

```bash
python tests/test_serial.py
mpirun -n 4 python tests/test_mpi_components.py --pr 2 --pc 2
mpirun -n 4 python tests/test_mpi_als.py --pr 2 --pc 2 --max_iter 5
```

## 🚀 Ejecución

Ejemplo local/MPI:

```bash
mpirun -n 4 python src/nmf_distributed.py \
  --pr 2 --pc 2 \
  --m 20000 --n 5000 --k 20 \
  --max_iter 20
```

Ejecutar campaña SLURM:

```bash
bash slurm/run_experiments.sh
```

## 📊 Resultados

Cada job guarda un CSV individual en:

```text
results/jobs/
```

Para construir la tabla final:

```bash
python results/aggregate_results.py \
  --jobs-dir results/jobs \
  --output results/nmf_results.csv
```

## 🧪 Configuraciones Evaluadas

Se evalúan las grillas:

```text
1x1, 2x1, 1x2, 2x2, 4x1, 1x4
```

con `T = 1, 2, 4` threads NumPy, más comparación NUMA con:

```text
--ntasks-per-socket=1
--ntasks-per-socket=2
```

## 📝

El solve de ALS usa una regularización diagonal pequeña para mejorar estabilidad numérica en las ecuaciones normales. Esta decisión mantiene la reproducibilidad experimental y evita fallos por matrices de Gram mal condicionadas.
