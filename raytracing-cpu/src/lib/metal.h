#ifndef METAL_H
#define METAL_H

#include "material.h"
#include "ray.h"
#include "hittable.h"

class Metal : public Material {
  public:
    Metal(const color &color, double fuzziness) : albedo(color), fuzziness(fuzziness) {}
    virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered) const override {
      vec3 reflected = unit(reflect(r.direction(), rec.normal)) ;
      scattered = ray(rec.p, reflected + fuzziness * random_in_unit_sphere());
      attentuation = albedo;
      return dot(scattered.direction(), rec.normal) > 0;
    }
  private:
    color albedo;
    double fuzziness;
};

#endif