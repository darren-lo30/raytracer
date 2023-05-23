#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <curand_kernel.h>
#include "vec3.h"

const static float infinity = std::numeric_limits<float>::infinity();
const static float pi = 3.1415926535;

// Utility Functions
__host__ __device__ static inline float degrees_to_radians(float degrees) {
  return degrees * pi / 180.0;
}

__host__ __device__ static inline float clamp(float x, float min, float max) {
  if(x < min) return min;
  if(x > max) return max;
  return x;
}

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

__host__ static unsigned char *get_char_array_from_color_array(color *colors, int size) {
  unsigned char *char_colors = new unsigned char[size * 3];
  for(int i = 0; i<size; ++i) {
    char_colors[i * 3 + 0] = static_cast<unsigned char>(colors[i].x() * 255);
    char_colors[i * 3 + 1] = static_cast<unsigned char>(colors[i].y() * 255);;
    char_colors[i * 3 + 2] = static_cast<unsigned char>(colors[i].z() * 255);;
  }

  return char_colors;
}