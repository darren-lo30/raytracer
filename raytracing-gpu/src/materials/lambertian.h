#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"
#include "../lib/vec3.h"
#include "../objects/hittable.h"
#include "../lib/utils.h"

class Lambertian : public Material {
  public: 
    __host__ __device__ Lambertian(const color &c);
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override;
  private:
    color albedo;
};
#endif