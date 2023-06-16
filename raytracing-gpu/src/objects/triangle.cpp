#include "triangle.h"

__host__ __device__ Triangle::Triangle() {}
__host__ __device__ Triangle::Triangle(point3 p1, point3 p2, point3 p3, Material *mat) : p1(p1), p2(p2), p3(p3), mat(mat) {}

__device__ bool Triangle::hit(const ray &r, float t_min, float t_max, HitRecord &rec) const {
  const static float epsilon = 0.00001;
  vec3 D = r.direction();
  vec3 e1 = p2 - p1;
  vec3 e2 = p3 - p1;

  vec3 h = cross(D, e2); // r cross e1
  float d = dot(e1, h);

  if(d < epsilon && d > -epsilon) return false; // Parallel to triangle

  float s = 1.0/d;
  vec3 K = r.origin() - p1;

  float u = s * dot(K, h);
  if(u > 1.0f || u < 0.0f) return false;

  vec3 D_cross_K = cross(D, K);
  float v = s * dot(e1, D_cross_K);
  if(u + v > 1.0f || v < 0.0f) return false;

  vec3 e1_cross_e2 = cross(e1, e2);
  float t = dot(K, e1_cross_e2) * s;

  if(t > t_max || t < t_min) return false;

  rec.t = t;
  rec.set_face_normal(r, e1_cross_e2);
  rec.mat = mat;
  return true;
}

__device__ void Triangle::setMat(Material *mat) {
  this->mat = mat;
}
