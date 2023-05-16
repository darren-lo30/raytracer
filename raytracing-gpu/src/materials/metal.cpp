#include "metal.h"
#include "../lib/utils.h"
#include "../lib/hittable.h"

__device__ Metal::Metal(const color &color, float fuzziness) :  albedo(color), fuzziness(fuzziness) {};
__device__ bool Metal::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered , curandState* state) const {
  vec3 reflected = unit(reflect(r.direction(), rec.normal)) ;
  scattered = ray(rec.p, reflected + fuzziness * random_in_unit_sphere(state));
  attentuation = albedo;
  return dot(scattered.direction(), rec.normal) > 0;
}