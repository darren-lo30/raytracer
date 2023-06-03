#pragma once

#include "../math/vec3.h"
#include "../math/vec2.h"
#include "triangle.h"
#include "hittable_list.h"
#include "thrust/device_vector.h"


// Models and meshes used in the GPU
// A model needs to be loaded in the cpu first, then it will be copied into the gpu with these classes

class Mesh {
  public:
    Mesh();
    thrust::device_vector<Triangle> triangles;
};

class Model {
  public:
    Model();
    thrust::device_vector<Mesh> meshes;
};