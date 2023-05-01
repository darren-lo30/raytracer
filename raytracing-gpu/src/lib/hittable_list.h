#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;
using std::vector;

class HittableList : public Hittable {
  public:
    HittableList() {}
    HittableList(shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray& r, double t_min, double t_max, HitRecord& rec) const override;

  public:
    vector<shared_ptr<Hittable>> objects;
};

bool HittableList::hit(const ray& r, double t_min, double t_max, HitRecord& rec) const {
  HitRecord temp_rec;
  double closest_t = t_max;
  bool object_hit = false;

  for (const auto& object: objects) {
    if(object->hit(r, t_min, closest_t, temp_rec)) {
      object_hit = true;
      closest_t = temp_rec.t;
      rec = temp_rec;
    }    
  }

  return object_hit;
}

#endif
