#include "metal.h"
#include "../objects/hittable.h"
#include "../utils/rand.h"

__device__ Metal::Metal(const color &color, float fuzziness) :  albedo(color), fuzziness(fuzziness) {};
__device__ bool Metal::scatter(const ray& r, const HitRecord& rec, color& attentuation, ray& scattered , curandState* state) const {
  vec3 reflected = normalize(reflect(r.direction(), rec.normal)) ;
  scattered = ray(rec.p, reflected + fuzziness * randomInUnitSphere(state));
  attentuation = albedo;
  return dot(scattered.direction(), rec.normal) > 0;
}