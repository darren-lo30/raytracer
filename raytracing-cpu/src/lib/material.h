#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"
#include "ray.h"

struct HitRecord;

class Material {
  public:
    virtual bool scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered) const = 0;
};

#endif