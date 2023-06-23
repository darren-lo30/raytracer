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
  bool frontFace;

  __device__ inline void setFaceNormal(const ray &r, const vec3& outwardNormal) {
    frontFace = dot(r.direction(), outwardNormal) < 0; // Normal opposes r implies outward normal implies front face.
    normal = frontFace ? outwardNormal : -outwardNormal; 
  }
};

class Hittable {
  public:
    __host__ __device__ Hittable() {}
    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const = 0;
};