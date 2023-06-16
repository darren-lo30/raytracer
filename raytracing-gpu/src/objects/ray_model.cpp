#include "ray_model.h"
#include <unordered_map>
#include <thrust/device_malloc.h>
#include <iostream>

__host__ __device__ RayMesh::RayMesh() {}

__host__ __device__ RayMesh::RayMesh(int num_triangles, Triangle **triangles) : num_triangles(num_triangles), triangles(triangles) {}

__device__ bool RayMesh::hit(const ray &r, float t_min, float t_max, HitRecord &rec) const {
  bool hit = false;
  float closest_t = t_max;


  for (int i = 0; i<num_triangles; ++i) {
    HitRecord triangle_hit_rec;    
    if(triangles[i]->hit(r, t_min, t_max, triangle_hit_rec)) {
      hit = true;
      if(triangle_hit_rec.t < closest_t) {
        closest_t = triangle_hit_rec.t;
        rec = triangle_hit_rec;
      }
    }
  }  

  return hit;
}

__device__ bool RayModel::hit(const ray &r, float t_min, float t_max, HitRecord &rec) const {
  bool hit = false;
  float closest_t = t_max;
  for (int i = 0; i<num_meshes; ++i) {
    HitRecord mesh_hit_rec;
    if(meshes[i]->hit(r, t_min, t_max, mesh_hit_rec)) {
      hit = true;
      if(mesh_hit_rec.t < closest_t) {
        closest_t = mesh_hit_rec.t;
        rec = mesh_hit_rec;
      }
    }
  }  

  return hit;
}

__host__ __device__ RayModel::RayModel() {}

__host__ __device__ RayModel::RayModel(int num_meshes, RayMesh **meshes) : num_meshes(num_meshes), meshes(meshes) {}


__global__ void init_triangle(Triangle **triangle, point3 p1, point3 p2, point3 p3) {
  *triangle = new Triangle(p1, p2, p3, nullptr);
}

__global__ void init_ray_mesh(RayMesh **ray_mesh, int num_triangles, Triangle **triangles) {
  *ray_mesh = new RayMesh(num_triangles, triangles);
}


__global__ void initMesh(RayMesh **mesh) {
  *mesh = new RayMesh(0, nullptr);
}
RayMesh *from_mesh(const Mesh &mesh) {
  thrust::device_vector<Triangle*> triangles;
  triangles.reserve(mesh.triangles.size());  

  // Allocate copies of triangle on gpu
  for(const Triangle &triangle: mesh.triangles) {
    thrust::device_ptr<Triangle*> device_triangle = thrust::device_malloc<Triangle*>(1);
    init_triangle<<<1, 1>>>(thrust::raw_pointer_cast(device_triangle), triangle.p1, triangle.p2, triangle.p3);
    triangles.push_back((Triangle *) *device_triangle);
  }

  thrust::device_ptr<RayMesh*> device_mesh = thrust::device_malloc<RayMesh*>(1);
  init_ray_mesh<<<1, 1>>>(thrust::raw_pointer_cast(device_mesh), triangles.size(), thrust::raw_pointer_cast(&triangles[0]));
  checkCudaErrors(cudaDeviceSynchronize());

  return *device_mesh;
}

// Annoying thing, for virtual functions to work, we need a double pointer.
__global__ void init_ray_model(RayModel **ray_model, int num_meshes, RayMesh **ray_meshes) {
  *ray_model = new RayModel(num_meshes, ray_meshes);
  printf("%d\n", *ray_model);
}


RayModel *from_model(const Model &model) {
  thrust::device_vector<RayMesh*> ray_meshes;
  ray_meshes.reserve(model.meshes.size());

  for(const Mesh &mesh: model.meshes) {
    RayMesh *device_mesh = from_mesh(mesh);
    ray_meshes.push_back(device_mesh);
  }

  RayModel **device_model;
  checkCudaErrors(cudaMallocManaged(&device_model, sizeof(RayModel *)));
  checkCudaErrors(cudaDeviceSynchronize());
  init_ray_model<<<1, 1>>>(device_model, ray_meshes.size(), thrust::raw_pointer_cast(&ray_meshes[0]));

  RayModel *device_model_ptr;
  cudaMemcpy(&device_model_ptr, device_model, sizeof(RayModel *), cudaMemcpyDeviceToHost);
  return device_model_ptr;
}
