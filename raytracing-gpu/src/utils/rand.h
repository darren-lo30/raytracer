#pragma once

#include <memory>
#include <random>
#include <curand_kernel.h>
#include "../math/vec3.h"


__device__ static inline float randomFloat(curandState *state) {
  return curand_uniform(state);
}

__device__ static inline float randomFloat(curandState *state, float min, float max) {
  return min + randomFloat(state) * (max - min);
}

__device__ static inline vec3 randomVec3(curandState *state) {
  return vec3(randomFloat(state), randomFloat(state), randomFloat(state));
}

__device__ static inline vec3 randomVec3(curandState *state, float min, float max) {
  return vec3(randomFloat(state, min, max), randomFloat(state, min, max), randomFloat(state, min, max));
}

__device__ static inline vec3 randomInUnitSphere(curandState *state) {
  vec3 rand;
  do {
    rand = randomVec3(state, -1, 1);
  } while(rand.lengthSquared() > 1);

  return rand;
}

__device__ static vec3 randomUnitVector(curandState *state) {
  return normalize(randomInUnitSphere(state));
}

__device__ static vec3 randomUnitInCircle(curandState *state) {
  vec3 rand;
  do {
    rand = vec3(randomFloat(state, -1, 1), randomFloat(state, -1, 1), 0);
  } while(rand.lengthSquared() >= 1);

  return rand;
}