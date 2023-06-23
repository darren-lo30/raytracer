#include "ray_model.h"
#include <unordered_map>
#include <thrust/device_malloc.h>
#include <iostream>

__host__ __device__ RayMesh::RayMesh() {}

__host__ __device__ RayMesh::RayMesh(int numTriangles, Triangle **triangles) : numTriangles(numTriangles), triangles(triangles) {}

__device__ bool RayMesh::hit(const ray &r, float tMin, float tMax, HitRecord &rec) const {
  bool hit = false;
  float closestT = tMax;


  for (int i = 0; i<numTriangles; ++i) {
    HitRecord triangleHitRec;    
    if(triangles[i]->hit(r, tMin, closestT, triangleHitRec)) {
      hit = true;
      closestT = triangleHitRec.t;
      rec = triangleHitRec;
    }
  }  

  return hit;
}

__device__ bool RayModel::hit(const ray &r, float tMin, float tMax, HitRecord &rec) const {
  bool hit = false;
  float closestT = tMax;
  for (int i = 0; i<numMeshes; ++i) {
    HitRecord meshHitRec;
    if(meshes[i]->hit(r, tMin, closestT, meshHitRec)) {
      hit = true;
      closestT = meshHitRec.t;
      rec = meshHitRec;
    }
  }  

  return hit;
}

__host__ __device__ RayModel::RayModel() {}

__host__ __device__ RayModel::RayModel(int numMeshes, RayMesh **meshes) : numMeshes(numMeshes), meshes(meshes) {}


__global__ void initTriangle(Triangle **triangle, point3 p1, point3 p2, point3 p3) {
  *triangle = new Triangle(p1, p2, p3, nullptr);
}

__global__ void initRayMesh(RayMesh **rayMesh, int numTriangles, Triangle **triangles) {
  *rayMesh = new RayMesh(numTriangles, triangles);
}


__global__ void initMesh(RayMesh **mesh) {
  *mesh = new RayMesh(0, nullptr);
}
RayMesh *fromMesh(const Mesh &mesh) {
  thrust::device_vector<Triangle*> triangles;
  triangles.reserve(mesh.triangles.size());  

  // Allocate copies of triangle on gpu
  for(const Triangle &triangle: mesh.triangles) {
    thrust::device_ptr<Triangle*> deviceTriangle = thrust::device_malloc<Triangle*>(1);
    initTriangle<<<1, 1>>>(thrust::raw_pointer_cast(deviceTriangle), triangle.p1, triangle.p2, triangle.p3);
    triangles.push_back((Triangle *) *deviceTriangle);
  }

  thrust::device_ptr<RayMesh*> deviceMeshes = thrust::device_malloc<RayMesh*>(1);
  initRayMesh<<<1, 1>>>(thrust::raw_pointer_cast(deviceMeshes), triangles.size(), thrust::raw_pointer_cast(&triangles[0]));
  checkCudaErrors(cudaDeviceSynchronize());

  return *deviceMeshes;
}

// Annoying thing, for virtual functions to work, we need a double pointer.
__global__ void initRayModel(RayModel **rayModel, int numMeshes, RayMesh **rayMeshes) {
  *rayModel = new RayModel(numMeshes, rayMeshes);
}


RayModel *fromModel(const Model &model) {
  thrust::device_vector<RayMesh*> rayMeshes;
  rayMeshes.reserve(model.meshes.size());

  for(const Mesh &mesh: model.meshes) {
    RayMesh *deviceMeshes = fromMesh(mesh);
    rayMeshes.push_back(deviceMeshes);
  }

  RayModel **deviceModel;
  checkCudaErrors(cudaMallocManaged(&deviceModel, sizeof(RayModel *)));
  checkCudaErrors(cudaDeviceSynchronize());
  initRayModel<<<1, 1>>>(deviceModel, rayMeshes.size(), thrust::raw_pointer_cast(&rayMeshes[0]));

  RayModel *deviceModelPtr;
  cudaMemcpy(&deviceModelPtr, deviceModel, sizeof(RayModel *), cudaMemcpyDeviceToHost);
  return deviceModelPtr;
}
