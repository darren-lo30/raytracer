#pragma once

#include "hittable.h"
#include "../math/vec3.h"
#include "../materials/material.h"

class Sphere : public Hittable { 
  public: 
    __host__ __device__ Sphere();
    __host__ __device__ Sphere(point3 center, float radius, Material* mat);

    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const override;
  private:
    point3 center;
    float radius;
    Material *mat;

};
