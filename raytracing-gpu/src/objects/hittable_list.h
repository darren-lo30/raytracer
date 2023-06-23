#pragma once

#include "hittable.h"

#include <memory>
#include <vector>

class HittableList : public Hittable {
  public:
    __host__ __device__ HittableList() { 
      idx = 0;
    }
    
    __host__ __device__ HittableList(int size, Hittable **objects) {
      idx = 0;
      this->size = size;
      this->objects = objects;
    }

    __host__ __device__ bool add(Hittable *object) {
      if(idx < size) {
        objects[idx++] = object;
        return true;
      }

      return false;
    }

    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const override {
      HitRecord tempRec;
      float closestT = tMax;
      bool objectHit = false;

      for (int i = 0; i<size; ++i) {
        if(objects[i] && objects[i]->hit(r, tMin, closestT, tempRec)) {
          objectHit = true;
          closestT = tempRec.t;
          rec = tempRec;
        }    
      }

      return objectHit;
    }

    int idx;
    int size;
    Hittable **objects;
};
