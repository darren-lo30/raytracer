#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"

class Camera {
  public:
    Camera(double viewport_width, double viewport_height, double focal_length) : 
      viewport_width(viewport_width),
      viewport_height(viewport_height),
      focal_length(focal_length) {}

    vec3 horizontal() {
      return vec3(viewport_width, 0, 0);
    }

    vec3 vertical() {
      return vec3(0, viewport_height, 0);
    }

    point3 lower_left_corner() {
      return position - horizontal()/2 - vertical()/2 - vec3(0, 0, focal_length);
    }
    point3 position;
  private: 
    double viewport_height;
    double viewport_width;
    double focal_length;
    
};

#endif