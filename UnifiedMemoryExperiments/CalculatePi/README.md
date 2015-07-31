### Unified Memory Experiment - Calculate Pi

This experiment has two goals: first to write a function that can be used as a CUDA kernel, function that uses openmp or a serial version and the second goal is to do a performance comparisson between kernel calls and reduce operations when using unified memory (cudaMallocManaged) and GPU memory (cudaMalloc). A side goal is to reuse as much code as possible and to be able to switch between CPU and GPU implementations by just changing one parameter in the CMakeLists file.

#### Setup
```
git clone https://github.com/felipegb94/HPC-Prototyping.git
cd HPC-Prototyping/UnifiedMemoryExperiments/CalculatePi/
mkdir /PATH/TO/DIRECTORY/WHERE/YOU/WANT/TOBUILD/build
cd /PATH/TO/DIRECTORY/WHERE/YOU/WANT/TOBUILD/build
```
If you are in euler (SBEL cluster) or a cluster in which cuda is not loaded then run:

```
module load cuda
```

If you are running this locally just make sure your workstation has an Nvidia GPU and also that you have CUDA and CUDA-SDK installed.

```
ccmake PATH/TO/HPC-Prototyping/UnifiedMemoryExperiments/CalculatePi/
make
```

#### Understanding the code

The code is not very long and is well documented. Stepping through it should be relatively trivial and as long as you read the comments you should be able to understand it. 

For an extensive analysis of this code please refer to the following technical report: 

ADD LINK TO TECHNICAL REPORT.
#### Usage
In order to choose if you want to run the serial, CPU parallel or GPU parallel version of the code modify the following lines at the top of CMakeLists.txt:

```
set(CUDA_ENABLED 1)
set(OPENMP_ENABLED 0)
```

The following combinations lead to use of GPU implementation:

```
CUDA_ENABLED 1
OPENMP_ENABLED 0

OR

CUDA_ENABLED 1
OPENMP_ENABLED 1
```

The following combinations lead to use of OpenMP implementation:

```
set(CUDA_ENABLED 0)
set(OPENMP_ENABLED 1)
```

The following combinations lead to use of serial implementation:

```
set(CUDA_ENABLED 0)
set(OPENMP_ENABLED 0)
```









