#include "diffuse_light.h"

__host__ __device__ DiffuseLight::DiffuseLight(const color &c): albedo(c) {}
__device__ bool DiffuseLight::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const {
  return false;
}
  
__device__ color DiffuseLight::emit(const HitRecord& rec) const {
  return albedo;
}