#ifndef COLOR_H
#define COLOR_H

#include <iostream> 
#include "vec3.h"

double to_gamma_space(double val) {
  return sqrt(val);
}
void draw_color(std::ostream &out, color pixel_color, int num_samples) {
  double r = clamp(to_gamma_space(pixel_color.x() / num_samples), 0, 0.999); 
  double g = clamp(to_gamma_space(pixel_color.y() / num_samples), 0, 0.999); 
  double b = clamp(to_gamma_space(pixel_color.z() / num_samples), 0, 0.999); 

  int rVal = static_cast<int>(r * 256);
  int gVal = static_cast<int>(g * 256);
  int bVal = static_cast<int>(b * 256);

  out << rVal << ' ' << gVal << ' ' << bVal << '\n';
}
#endif