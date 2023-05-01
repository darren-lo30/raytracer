#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec3.h"
#include "ray.h"

class Material;

struct HitRecord {
  point3 p;
  vec3 normal;
  std::shared_ptr<Material> mat; 
  double t;
  bool front_face;

  inline void set_face_normal(const ray &r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0; // Normal opposes r implies outward normal implies front face.
    normal = front_face ? outward_normal : -outward_normal; 
  }
};

class Hittable {
  public:
    virtual bool hit(const ray &r, double t_min, double t_max, HitRecord &rec) const = 0;
};

#endif