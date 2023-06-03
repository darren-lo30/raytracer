#pragma once

#include "vec3.h"

class ray {
  public: 
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& dir) : orig(origin), dir(dir) {}

    __host__ __device__ point3 origin() const { return orig; };
    __host__ __device__ vec3 direction() const { return dir; };

    __host__ __device__ point3 at(const float &t) const { return orig + t * dir; };
  private:
    point3 orig;
    vec3 dir;
};
