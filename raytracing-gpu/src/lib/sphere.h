#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include "material.h"

class Sphere : public Hittable { 
  public: 
    Sphere() {}
    Sphere(point3 center, double radius, shared_ptr<Material> mat) : center(center), radius(radius), mat(mat) {}

    virtual bool hit(
      const ray &r, double t_min, double t_max, HitRecord &rec
    ) const override {
      vec3 oc = r.origin() - center;
      double a = r.direction().length_squared();
      double half_b = dot(r.direction(), oc);
      double c = oc.length_squared() - radius * radius;
      double discriminant = half_b * half_b - a*c;

      if(discriminant < 0) {
        return false;
      } 

      double sqrtd = sqrt(discriminant);
      

      double t = (-half_b - sqrtd) / a;
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
    
  private:
    point3 center;
    double radius;
    shared_ptr<Material> mat;

};
#endif