#include "sphere.h"

__host__ __device__ Sphere::Sphere() {}
__host__ __device__ Sphere::Sphere(point3 center, float radius, Material* mat) : center(center), radius(radius), mat(mat) {}

__device__ bool Sphere::hit(const ray &r, float tMin, float tMax, HitRecord &rec) const {
  vec3 oc = r.origin() - center;
  float a = r.direction().lengthSquared();
  float halfB = dot(r.direction(), oc);
  float c = oc.lengthSquared() - radius * radius;
  float discriminant = halfB * halfB - a*c;

  if(discriminant < 0) {
    return false;
  } 

  float sqrtd = sqrt(discriminant);
  

  float t = (-halfB - sqrtd) / a;
  if(t < tMin || t > tMax) {
    t = (-halfB + sqrtd) / a;
    if(t < tMin || t > tMax) return false;
  }

  rec.t = t;
  rec.p = r.at(t);
  vec3 outwardNormal = (rec.p - center) / radius;
  rec.setFaceNormal(r, outwardNormal);
  rec.mat = mat;

  return true;
}