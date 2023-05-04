#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"
#include "ray.h"
#include <curand_kernel.h>

struct HitRecord;

class Material {
  public:
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered , curandState* state) const = 0;

    __device__ virtual bool test() const { 
      return true; 
    }
};

#endif