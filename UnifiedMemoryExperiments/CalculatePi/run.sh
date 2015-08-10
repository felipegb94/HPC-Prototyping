#!/bin/sh
cd build

echo 1
export OMP_NUM_THREADS=1
time ./pi

echo 2
export OMP_NUM_THREADS=2
time ./pi

echo 4
export OMP_NUM_THREADS=4
time ./pi

echo 8
export OMP_NUM_THREADS=8
time ./pi

echo 16
export OMP_NUM_THREADS=16
time ./pi

echo 20
export OMP_NUM_THREADS=20
time ./pi

echo 32
export OMP_NUM_THREADS=32
time ./pi

echo 40
export OMP_NUM_THREADS=40
time ./pi

