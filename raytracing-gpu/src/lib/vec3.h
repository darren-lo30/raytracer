#pragma once

#include <cmath>
#include <iostream>

class vec3 {
  public:
    __host__ __device__ vec3() : e{0, 0, 0} {};
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {};

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); };
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }


    __host__ __device__ vec3& operator+=(const vec3 &v) {
      e[0] += v[0];
      e[1] += v[1];
      e[2] += v[2];
      return *this;
    }

    __host__ __device__ vec3& operator-=(const vec3 &v) {
      *this+=-v;
      return *this;
    }

    __host__ __device__ vec3& operator*=(const float s) {
      e[0] *= s;
      e[1] *= s;
      e[2] *= s;
      return *this;
    }

    __host__ __device__ vec3& operator/=(const float s) {
      e[0] /= s;
      e[1] /= s;
      e[2] /= s;
      return *this;
    }

    __host__ __device__ float length_squared() const {
      return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    
    __host__ __device__ float length() const {
      return sqrt(length_squared());
    }

    __host__ __device__ bool near_zero() {
      float epsilon = 1e-8;
      return (e[0] < epsilon) && (e[1] < epsilon) && (e[2] < epsilon);
    }
  private:
    float e[3];
};

using point3 = vec3;
using color = vec3;

static inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
  return out << v.x() << " " << v.y() << " " << v.z();
}

__host__ __device__ static inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ static inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ static inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ static inline vec3 operator*(const float &s, const vec3 &u) {
  return vec3(u.x() * s, u.y() * s, u.z() * s);
}

__host__ __device__ static inline vec3 operator/(const vec3 &u, const float &s) {
  return vec3(u.x() / s, u.y() / s, u.z() / s);
}

__host__ __device__ static inline float dot(const vec3 &u, const vec3 &v) {
  return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

__host__ __device__ static inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(
    u.y()*v.z() - u.z()*v.y(),
    u.z()*v.x() - u.x()*v.z(),
    u.x()*v.y() - u.y()*v.x());
}

__host__ __device__ static inline vec3 reflect(const vec3 &v, const vec3 &axis) {
  return v - 2*dot(v, axis)*axis;
}

__host__ __device__ static inline vec3 unit(const vec3 &u) {
  return u / u.length();
}

__host__ __device__ static inline vec3 refract(const vec3& unit_v, const vec3& n, float etai_over_etat) {
  float cos_theta = dot(-unit_v, n);
  vec3 r_perpendicular = etai_over_etat * (unit_v + cos_theta * n);
  vec3 r_parallel = -sqrt(1 - r_perpendicular.length_squared()) * n;

  return r_perpendicular + r_parallel;
}
