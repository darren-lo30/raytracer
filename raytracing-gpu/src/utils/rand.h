#pragma once

#include <memory>
#include <random>
#include <curand_kernel.h>
#include "../math/vec3.h"


__device__ static inline float random_float(curandState *state) {
  return curand_uniform(state);
}

__device__ static inline float random_float(curandState *state, float min, float max) {
  return min + random_float(state) * (max - min);
}

__device__ static inline vec3 random_vec3(curandState *state) {
  return vec3(random_float(state), random_float(state), random_float(state));
}

__device__ static inline vec3 random_vec3(curandState *state, float min, float max) {
  return vec3(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
}

__device__ static inline vec3 random_in_unit_sphere(curandState *state) {
  vec3 rand;
  do {
    rand = random_vec3(state, -1, 1);
  } while(rand.length_squared() > 1);

  return rand;
}

__device__ static vec3 random_unit_vector(curandState *state) {
  return unit(random_in_unit_sphere(state));
}

__device__ static vec3 random_in_unit_circle(curandState *state) {
  vec3 rand;
  do {
    rand = vec3(random_float(state, -1, 1), random_float(state, -1, 1), 0);
  } while(rand.length_squared() >= 1);

  return rand;
}