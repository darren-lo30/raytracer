#pragma once

#include "../math/vec3.h"
#include "../math/ray.h"
#include <curand_kernel.h>

class HitRecord;

class Material {
  public:
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered , curandState* state) const = 0;
    __device__ virtual color emit(const HitRecord& rec) const {
      return color(0, 0, 0);
    };
};