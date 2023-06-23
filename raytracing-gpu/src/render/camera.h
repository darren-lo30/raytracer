#pragma once

#include "../math/vec3.h"
#include "../math/ray.h"
#include "../utils/managed.h"
#include "curand_kernel.h"
#include "../utils/utils.h"

class Camera : public Managed {
  public:
    __device__ __host__ Camera(point3 lookFrom, point3 lookAt, vec3 viewUp, float fovDeg, float aspectRatio, float aperture, float focusDist) {
      float fovRad = degreesToRadians(fovDeg);
      front = normalize(lookAt - lookFrom);
      right = normalize(cross(front, viewUp));
      up = cross(right, front);

      float h = tan(fovRad/2) * focusDist;
      float viewportHeight = h*2;
      float viewportWidth = aspectRatio * viewportHeight;
      

      position = lookFrom;
      horizontal = viewportWidth * right;
      vertical = viewportHeight * up;
      lowerLeftCorner = position + focusDist * front - horizontal / 2 - vertical / 2;

      lensRadius = aperture / 2;
    }

    __device__ ray getRay(float h, float v, curandState* state) const {
      vec3 offset = vec3(0, 0, 0);
      // vec3 offset = lensRadius * random_in_unit_circle(state);
      vec3 rayOrigin = position + offset.x() * right + offset.y() * up;
      return ray(rayOrigin, lowerLeftCorner + h * horizontal + v * vertical - rayOrigin);
    }

  private: 
    point3 position;
    point3 lowerLeftCorner;
    vec3 horizontal, vertical;
    vec3 front, right, up;
    float lensRadius;
};
