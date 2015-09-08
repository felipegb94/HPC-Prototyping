#include <stdio.h>

__global__ void doGPUWork(int numData, int *data) {
  if (threadIdx.x < numData) {
    data[threadIdx.x] = threadIdx.x;
  } 
}

void sayHello(int *numDevices) {
  int numData = 2;
  int data[numData];
  int dev_data[numData];
  int i;

  cudaGetDeviceCount(numDevices);

  cudaMalloc((void**)&dev_data, numData);

  doGPUWork<<<1, numData>>>(numData, dev_data);

  cudaMemcpy(data, dev_data, numData, cudaMemcpyDeviceToHost);

  // BUGFIX: This should print 0, 1, etc., but does not yet
  for (i = 0; i < numData; i++) {
    printf("%d\n", data[i]);
  }
}
