# HPC-Prototyping
This repository contains various projects for testing HPC technologies and algorithms.

## CUDA Unified Memory Experiments
### Calculate Pi
This experiment has two goals: first to write a function that can be used as a CUDA kernel, function that uses openmp or a serial version and the second goal is to do a performance comparisson between kernel calls and reduce operations when using unified memory (cudaMallocManaged) and GPU memory (cudaMalloc). A side goal is to reuse as much code as possible and to be able to switch between CPU and GPU implementations by just changing one parameter in the CMakeLists file.


## Multi GPU Experiments
### Single Node Multiple GPU 
### Multiple Nodes Multiple GPU

