#pragma once

#include <cmath>
#include <iostream>

class vec2 {
  public:
    __host__ __device__ vec2() : e{0, 0} {}
    __host__ __device__ vec2(float e0, float e1) : e{e0, e1} {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }

    __host__ __device__ vec2 operator-() const { return vec2(-e[0], -e[1]); };
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec2& operator+=(const vec2 &v) {
      e[0] += v[0];
      e[1] += v[1];
      return *this;
    }

    __host__ __device__ vec2& operator-=(const vec2 &v) {
      *this+=-v;
      return *this;
    }

    __host__ __device__ vec2& operator*=(const float s) {
      e[0] *= s;
      e[1] *= s;
      return *this;
    }

    __host__ __device__ vec2& operator/=(const float s) {
      e[0] /= s;
      e[1] /= s;
      return *this;
    }

    __host__ __device__ float length_squared() const {
      return e[0]*e[0] + e[1]*e[1];
    }
    
    __host__ __device__ float length() const {
      return sqrt(length_squared());
    }

    __host__ __device__ bool near_zero() {
      float epsilon = 1e-8;
      return (e[0] < epsilon) && (e[1] < epsilon);
    }
  private:
    float e[2];
};


static inline std::ostream& operator<<(std::ostream &out, const vec2 &v) {
  return out << v.x() << " " << v.y();
}

__host__ __device__ static inline vec2 operator+(const vec2 &u, const vec2 &v) {
  return vec2(u.x() + v.x(), u.y() + v.y());
}

__host__ __device__ static inline vec2 operator-(const vec2 &u, const vec2 &v) {
  return vec2(u.x() - v.x(), u.y() - v.y());
}

__host__ __device__ static inline vec2 operator*(const vec2 &u, const vec2 &v) {
  return vec2(u.x() * v.x(), u.y() * v.y());
}

__host__ __device__ static inline vec2 operator*(const float &s, const vec2 &u) {
  return vec2(u.x() * s, u.y() * s);
}

__host__ __device__ static inline vec2 operator/(const vec2 &u, const float &s) {
  return vec2(u.x() / s, u.y() / s);
}


