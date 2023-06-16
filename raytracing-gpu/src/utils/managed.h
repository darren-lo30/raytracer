#pragma once

#include "error.h"

// Inherit to initialize on both device and host. Does not support classes with virtual functions.
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