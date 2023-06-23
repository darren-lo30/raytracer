#pragma once

#include "constants.h"

// Utility Functions
__host__ __device__ static inline float degreesToRadians(float degrees) {
  return degrees * pi / 180.0;
}

__host__ __device__ static inline float clamp(float x, float min, float max) {
  if(x < min) return min;
  if(x > max) return max;
  return x;
}
