#pragma once

#include "material.h"
#include "../math/vec3.h"
#include "../objects/hittable.h"

class Lambertian : public Material {
  public: 
    __host__ __device__ Lambertian(const color &c);
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override;
  private:
    color albedo;
};
