#pragma once

#include "hittable.h"
#include "../math/vec3.h"
#include "../materials/material.h"

class Triangle : public Hittable { 
  public: 
    __host__ __device__ Triangle();
    __host__ __device__ Triangle(point3 p1, point3 p2, point3 p3, Material *mat);

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, HitRecord &rec) const override;
    point3 p1, p2, p3;
  private:
    Material *mat;
};

static inline std::ostream& operator<<(std::ostream &out, const Triangle &t) {
  return out << t.p1 << " " << t.p2 << " " << t.p3;
}
