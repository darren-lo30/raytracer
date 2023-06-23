#pragma once

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )

#include <iostream>

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line);