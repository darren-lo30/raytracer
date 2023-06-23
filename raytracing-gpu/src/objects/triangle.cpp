#include "triangle.h"

__host__ __device__ Triangle::Triangle() {}
__host__ __device__ Triangle::Triangle(point3 p1, point3 p2, point3 p3, Material *mat) : p1(p1), p2(p2), p3(p3), mat(mat) {}

__device__ bool Triangle::hit(const ray &r, float tMin, float tMax, HitRecord &rec) const {
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

  vec3 DCrossK = cross(D, K);
  float v = s * dot(e1, DCrossK);
  if(u + v > 1.0f || v < 0.0f) return false;

  vec3 e1Cross2 = cross(e1, e2);
  float t = dot(K, e1Cross2) * s;

  if(t > tMax || t < tMin) return false;

  rec.t = t;
  rec.p = r.at(t);
  rec.setFaceNormal(r, e1Cross2);
  rec.mat = mat;
  return true;
}

__device__ void Triangle::setMat(Material *mat) {
  this->mat = mat;
}
