#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"
#include "vec3.h"
#include "hittable.h"

class Lambertian : public Material {
  public: 
    Lambertian(const color &c) : albedo(c) {};

    virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered) const override {
      vec3 scatter_direction = rec.normal + random_unit_vector();
      if(scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
      }

      scattered = ray(rec.p, scatter_direction);
      attentuation = albedo;
      return true;
    }
  private:
    color albedo;
};
#endif