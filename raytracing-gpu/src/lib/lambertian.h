#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"
#include "vec3.h"
#include "hittable.h"

class Lambertian : public Material {
  public: 
    __host__ __device__ Lambertian(const color &c) : albedo(c) {};

    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override {
      vec3 scatter_direction = rec.normal + random_unit_vector(state);
      if(scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
      }

      scattered = ray(rec.p, scatter_direction);
      attentuation = albedo;
      return true;
    }

    __device__ virtual bool test() const override {
      return false;
    }
  private:
    color albedo;
};
#endif