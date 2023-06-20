#pragma once

#include "vec3.h"

class ray {
  public: 
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& dir) : orig(origin), dir(dir) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }

    __host__ __device__ point3 at(const float &t) const { return orig + t * dir; }

    __host__ __device__ static ray null_ray() {
      return ray(point3(0, 0,0), vec3(0, 0, 0));
    }

    __host__ __device__ bool is_null_ray() const {
      return orig.near_zero() && dir.near_zero();
    }
  private:
    point3 orig;
    vec3 dir;
};
