#ifndef METAL_H
#define METAL_H

#include "material.h"

class Metal : public Material {
  public:
    __host__ __device__ Metal(const color &color, float fuzziness);
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override;
  private:
    color albedo;
    float fuzziness;
};

#endif