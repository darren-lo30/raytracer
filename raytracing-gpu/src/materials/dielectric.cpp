#include "dielectric.h"
#include "../utils/rand.h"

__host__ __device__ Dielectric::Dielectric(float indexOfRefraction) : indexOfRefraction(indexOfRefraction) {}
__device__ bool Dielectric::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState *state) const {
  attentuation = color(1, 1, 1);
  float refractionRatio = rec.frontFace ? (1.0)/indexOfRefraction : indexOfRefraction;

  vec3 unitDirection = normalize(r.direction());

  float cosTheta = dot(-unitDirection, rec.normal);
  float sinTheta = sqrt(1 - cosTheta * cosTheta);

  vec3 outDirection;
  if(refractionRatio * sinTheta > 1.0 || reflectance(cosTheta, indexOfRefraction) > randomFloat(state)) {
    outDirection = reflect(unitDirection, rec.normal);
  } else {
    outDirection = refract(unitDirection, rec.normal, refractionRatio);
  }

  scattered = ray(rec.p, outDirection);
  return true;
}