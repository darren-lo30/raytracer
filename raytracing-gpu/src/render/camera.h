#ifndef CAMERA_H
#define CAMERA_H

#include "../lib/vec3.h"
#include "../lib/ray.h"
#include "../lib/managed.h"
#include "curand_kernel.h"
#include "../lib/utils.h"

class Camera : public Managed {
  public:
    __device__ __host__ Camera(point3 look_from, point3 look_at, vec3 view_up, float fov_deg, float aspect_ratio, float aperture, float focus_dist) {
      float fov_rad = degrees_to_radians(fov_deg);
      front = unit(look_at - look_from);
      right = unit(cross(front, view_up));
      up = cross(right, front);

      float h = tan(fov_rad/2) * focus_dist;
      float viewport_height = h*2;
      float viewport_width = aspect_ratio * viewport_height;
      

      position = look_from;
      horizontal = viewport_width * right;
      vertical = viewport_height * up;
      lower_left_corner = position + focus_dist * front - horizontal / 2 - vertical / 2;

      lens_radius = aperture / 2;
    }

    __device__ ray get_ray(float h, float v, curandState* state) const {
      vec3 offset = vec3(0, 0, 0);
      // vec3 offset = lens_radius * random_in_unit_circle(state);
      vec3 ray_origin = position + offset.x() * right + offset.y() * up;
      return ray(ray_origin, lower_left_corner + h * horizontal + v * vertical - ray_origin);
    }

  private: 
    point3 position;
    point3 lower_left_corner;
    vec3 horizontal, vertical;
    vec3 front, right, up;
    float lens_radius;
};

#endif