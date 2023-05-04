#ifndef MANAGED_H
#define MANAGED_H

#include "error.h"

class Managed {
  public:
    void *operator new(size_t len) {
      void *ptr;
      checkCudaErrors(cudaMallocManaged(&ptr, len));
      checkCudaErrors(cudaDeviceSynchronize());
      return ptr;
    }

    void operator delete(void *ptr) {
      cudaDeviceSynchronize();
      cudaFree(ptr);
    }

    
};


#endif