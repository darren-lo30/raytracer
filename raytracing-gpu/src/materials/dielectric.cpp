#include "dielectric.h"

__host__ __device__ Dielectric::Dielectric(float index_of_refraction) : index_of_refraction(index_of_refraction) {}
__device__ bool Dielectric::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState *state) const {
  attentuation = color(1, 1, 1);
  float refraction_ratio = rec.front_face ? (1.0)/index_of_refraction : index_of_refraction;

  vec3 unit_direction = unit(r.direction());

  float cos_theta = dot(-unit_direction, rec.normal);
  float sin_theta = sqrt(1 - cos_theta * cos_theta);

  vec3 out_direction;
  if(refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, index_of_refraction) > random_float(state)) {
    out_direction = reflect(unit_direction, rec.normal);
  } else {
    out_direction = refract(unit_direction, rec.normal, refraction_ratio);
  }

  scattered = ray(rec.p, out_direction);
  return true;
}