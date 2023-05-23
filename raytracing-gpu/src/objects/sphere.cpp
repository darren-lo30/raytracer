#include "sphere.h"

__host__ __device__ Sphere::Sphere() {}
__host__ __device__ Sphere::Sphere(point3 center, float radius, Material* mat) : center(center), radius(radius), mat(mat) {}

__device__ bool Sphere::hit(const ray &r, float t_min, float t_max, HitRecord &rec) const {
  vec3 oc = r.origin() - center;
  float a = r.direction().length_squared();
  float half_b = dot(r.direction(), oc);
  float c = oc.length_squared() - radius * radius;
  float discriminant = half_b * half_b - a*c;

  if(discriminant < 0) {
    return false;
  } 

  float sqrtd = sqrt(discriminant);
  

  float t = (-half_b - sqrtd) / a;
  if(t < t_min || t > t_max) {
    t = (-half_b + sqrtd) / a;
    if(t < t_min || t > t_max) return false;
  }

  rec.t = t;
  rec.p = r.at(t);
  vec3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
  rec.mat = mat;

  return true;
}