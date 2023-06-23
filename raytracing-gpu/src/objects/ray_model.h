#pragma once

#include "../math/vec3.h"
#include "../math/vec2.h"
#include "../math/ray.h"
#include "triangle.h"
#include "hittable.h"
#include "hittable_list.h"
#include "thrust/device_vector.h"
#include "../utils/managed.h"
#include "model.h"


// Models and meshes used in the GPU
// A model needs to be loaded in the cpu first, then it will be copied into the gpu with these classes

class RayMesh : public Hittable {
  public:
    __host__ __device__ RayMesh();
    __host__ __device__ RayMesh(int numTriangles, Triangle **triangles);
    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const override;
    
    int numTriangles;
    Triangle **triangles;
};

class RayModel : public Hittable {
  public:
    __host__ __device__ RayModel();
    __host__ __device__ RayModel(int numMeshes, RayMesh **meshes);
    __device__ virtual bool hit(const ray &r, float tMin, float tMax, HitRecord &rec) const override;

    int numMeshes;
    RayMesh **meshes;
};

RayMesh *fromMesh(const Mesh &model);
RayModel *fromModel(const Model &model);
