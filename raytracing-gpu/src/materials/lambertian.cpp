#include "lambertian.h"
#include "../utils/rand.h"

__host__ __device__ Lambertian::Lambertian(const color &c) : albedo(c) {};

__device__ bool Lambertian::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const {
  vec3 scatterDirection = rec.normal + randomUnitVector(state);
  if(scatterDirection.nearZero()) {
    scatterDirection = rec.normal;
  }

  scattered = ray(rec.p, scatterDirection);
  attentuation = albedo;
  return true;
}