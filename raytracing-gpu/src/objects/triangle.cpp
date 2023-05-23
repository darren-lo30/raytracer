#include "triangle.h"

__host__ __device__ Triangle::Triangle() {}
__host__ __device__ Triangle::Triangle(point3 p1, point3 p2, point3 p3) : p1(p1), p2(p2), p3(p3) {}

__device__ bool Triangle::hit(const ray &r, float t_min, float t_max, HitRecord &rec) const {
  return true;
}