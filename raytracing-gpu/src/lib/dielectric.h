#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "ray.h"
#include "hittable.h"
#include "vec3.h"

class Dielectric : public Material {
  public:
    __host__ __device__ Dielectric(float index_of_refraction) : index_of_refraction(index_of_refraction) {}
    __device__ virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered, curandState *state) const override {
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

  private:
    float index_of_refraction;

    __device__ static float reflectance(float cosine , float ir) {
      float r0 = (1 - ir) / (1 + ir);
      r0 = r0 * r0;
      return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif