#ifndef METAL_H
#define METAL_H

#include "material.h"
#include "ray.h"
#include "hittable.h"

class Metal : public Material {
  public:
    __host__ __device__ Metal(const color &color, float fuzziness) : albedo(color), fuzziness(fuzziness) {}
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState* state) const override {
      vec3 reflected = unit(reflect(r.direction(), rec.normal)) ;
      scattered = ray(rec.p, reflected + fuzziness * random_in_unit_sphere(state));
      attentuation = albedo;
      return dot(scattered.direction(), rec.normal) > 0;
    }
  private:
    color albedo;
    float fuzziness;
};

#endif