#include <stdio.h>
#include <mpi.h>
#include "hello.h"

int main(int argc, char **argv) {
  int rank = 0;
  int size = 1;
  int numDevices;
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Get_processor_name(name, &len);
  sayHello(&numDevices);

  printf("Processor: %s #CUDA devices: %d\n",
    name, numDevices);


  MPI_Finalize();

  return 0;
}
