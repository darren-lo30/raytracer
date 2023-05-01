#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

class Camera {
  public:
    Camera(point3 look_from, point3 look_at, vec3 view_up, double fov_deg, double aspect_ratio, double aperture, double focus_dist) {
      double fov_rad = degrees_to_radians(fov_deg);
      front = unit(look_at - look_from);
      right = unit(cross(front, view_up));
      up = cross(right, front);

      double h = tan(fov_rad/2) * focus_dist;
      double viewport_height = h*2;
      double viewport_width = aspect_ratio * viewport_height;
      

      position = look_from;
      horizontal = viewport_width * right;
      vertical = viewport_height * up;
      lower_left_corner = position + focus_dist * front - horizontal / 2 - vertical / 2;

      lens_radius = aperture / 2;
    }

    ray get_ray(double h, double v) {
      vec3 offset = lens_radius * random_in_unit_circle();
      vec3 ray_origin = position + offset.x() * right + offset.y() * up; 
      return ray(ray_origin, lower_left_corner + h * horizontal + v * vertical - ray_origin);
    }

  private: 
    point3 position;
    point3 lower_left_corner;
    vec3 horizontal, vertical;
    vec3 front, right, up;
    double lens_radius;
};

#endif