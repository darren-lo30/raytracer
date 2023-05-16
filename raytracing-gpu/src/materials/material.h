#ifndef MATERIAL_H
#define MATERIAL_H

#include "../lib/vec3.h"
#include "../lib/ray.h"
#include <curand_kernel.h>

class HitRecord;

class Material {
  public:
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered , curandState* state) const = 0;
};

#endif