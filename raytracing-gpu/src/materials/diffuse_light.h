#pragma once

#include "material.h"
#include "../math/vec3.h"
#include "../objects/hittable.h"

class DiffuseLight : public Material {
  public: 
    __host__ __device__ DiffuseLight(const color &c);
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override;
    __device__ virtual color emit(const HitRecord& rec) const override;
  private:
    color albedo;
};
