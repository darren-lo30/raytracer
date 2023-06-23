#pragma once

#include "hittable.h"
#include "../math/vec3.h"
#include "../materials/material.h"

class Triangle : public Hittable { 
  public: 
    __host__ __device__ Triangle();
    __host__ __device__ Triangle(point3 p1, point3 p2, point3 p3, Material *mat);

    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const override;
    __device__ void setMat(Material *mat);

    point3 p1, p2, p3;
  private:
    Material *mat;
};
