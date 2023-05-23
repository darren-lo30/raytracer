#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "../lib/ray.h"
#include "../objects/hittable.h"
#include "../lib/vec3.h"
#include "../lib/utils.h"

class Dielectric : public Material {
  public:
    __host__ __device__ Dielectric(float index_of_refraction);
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState *state) const override;
  private:
    float index_of_refraction;

    __device__ static float reflectance(float cosine , float ir) {
      float r0 = (1 - ir) / (1 + ir);
      r0 = r0 * r0;
      return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif