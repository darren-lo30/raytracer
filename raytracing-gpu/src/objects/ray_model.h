// #include "../lib/vec3.h"
// #include "../lib/vec2.h"
// #include "triangle.h"
// #include "hittable_list.h"
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>


// // Models and meshes used in the GPU
// // A model needs to be loaded in the cpu first, then it will be copied into the gpu with these classes

// class RayMesh : private HittableList {
//   public:
//     RayMesh();
//     RayMesh(Triangle **mesh_triangles, int num_triangles);
// };

// class RayModel : private HittableList {
//   public:
//     RayModel();
//     RayModel(Mesh **meshes, int num_meshes);
// };