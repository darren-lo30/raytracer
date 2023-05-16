#include "lambertian.h"

__host__ __device__ Lambertian::Lambertian(const color &c) : albedo(c) {};

__device__ bool Lambertian::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const {
  vec3 scatter_direction = rec.normal + random_unit_vector(state);
  if(scatter_direction.near_zero()) {
    scatter_direction = rec.normal;
  }

  scattered = ray(rec.p, scatter_direction);
  attentuation = albedo;
  return true;
}