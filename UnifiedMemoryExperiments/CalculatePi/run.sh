#!/bin/sh
export CUDA_ENABLED=1
export OPENMP_ENABLED=0
export NUM_RECTS=10000
make



# export CUDA_ENABLED=0
# export OPENMP_ENABLED=1
# echo 1
# export OMP_NUM_THREADS=1
# time ./piUniMem

# echo 2
# export OMP_NUM_THREADS=2
# time ./piUniMem

# echo 4
# export OMP_NUM_THREADS=4
# time ./piUniMem

# echo 8
# export OMP_NUM_THREADS=8
# time ./piUniMem

# echo 16
# export OMP_NUM_THREADS=16
# time ./piUniMem

# echo 20
# export OMP_NUM_THREADS=20
# time ./piUniMem

# echo 32
# export OMP_NUM_THREADS=32
# time ./piUniMem

# echo 40
# export OMP_NUM_THREADS=40
# time ./piUniMem
