#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

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

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, HitRecord &rec) const override {
      HitRecord temp_rec;
      float closest_t = t_max;
      bool object_hit = false;

      for (int i = 0; i<size; ++i) {
        if(objects[i] && objects[i]->hit(r, t_min, closest_t, temp_rec)) {
          object_hit = true;
          closest_t = temp_rec.t;
          rec = temp_rec;
        }    
      }

      return object_hit;
    }

    int idx;
    int size;
    Hittable **objects;
};


#endif
