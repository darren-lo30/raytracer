#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <limits>
#include <memory>
#include <random>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}

inline double random_double() {
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}

inline double random_double(double min, double max) {
  return min + random_double() * (max - min);
}

inline double clamp(double x, double min, double max) {
  if(x < min) return min;
  if(x > max) return max;
  return x;
}

inline double sign(double x) {
  if(x > 0) return 1;
  if(x < 0) return -1;
  return 0;
}

#endif