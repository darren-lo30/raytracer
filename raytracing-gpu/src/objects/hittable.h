#pragma once 

#include "../math/vec3.h"
#include "../math/ray.h"
#include "../math/vec2.h"

class Material;

struct HitRecord {
  point3 p;
  vec3 normal;
  Material *mat; 
  vec2 uv;
  float t;
  bool front_face;

  __device__ inline void set_face_normal(const ray &r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0; // Normal opposes r implies outward normal implies front face.
    normal = front_face ? outward_normal : -outward_normal; 
  }
};

class Hittable {
  public:
    __host__ __device__ Hittable() {}
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};