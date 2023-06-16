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
    __host__ __device__ RayMesh(int num_triangles, Triangle **triangles);
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, HitRecord &rec) const override;
    
    int num_triangles;
    Triangle **triangles;
};

class RayModel : public Hittable {
  public:
    __host__ __device__ RayModel();
    __host__ __device__ RayModel(int num_meshes, RayMesh **meshes);
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, HitRecord &rec) const override;

    int num_meshes;
    RayMesh **meshes;
};

RayMesh *from_mesh(const Mesh &model);
RayModel *from_model(const Model &model);
