
#include <iostream>
#include <fstream>
#include "utils/utils.h"
#include "render/ray_renderer.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "render/camera.h"
#include "objects/hittable.h"
#include "objects/hittable_list.h"
#include "objects/sphere.h"
#include "materials/metal.h"
#include "materials/lambertian.h"
#include "materials/dielectric.h"
#include <curand_kernel.h>
#include "display/window.h"
#include "render/shader.h"
#include "objects/triangle.h"
#include "objects/model_loader.h"
#include <time.h>
#include "render/window_renderer.h"
#include "utils/rand.h"
#include "objects/ray_model.h"

__global__ void random_scene(HittableList** world, RayModel *ray_model) {
  if(threadIdx.x != 0 || blockIdx.x != 0) return;

  const int size = 25 * 25 + 20;
  Hittable** objects = new Hittable*[size];
  *world = new HittableList(size, objects);

  curandState state_val;
  curand_init(1984, 0, 0, &state_val);
  curandState *state = &state_val;


  auto ground_material = new Lambertian(color(0.5, 0.5, 0.5));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_float(state);
      point3 center(a + 0.9*random_float(state), 0.2, b + 0.9*random_float(state));
      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        Material* sphere_mat;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = random_vec3(state) * random_vec3(state);
          sphere_mat = new Lambertian(albedo);
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = random_vec3(state, 0.5, 1);
          auto fuzz = random_float(state, 0, 0.5);
          sphere_mat = new Metal(albedo, fuzz);
        } else {
          // glass
          sphere_mat = new Dielectric(1.5);
        }

        (*world)->add(new Sphere(center, 0.2, sphere_mat));
      }
    }
  }

  auto material1 = new Dielectric(1.5);
  (*world)->add(new Sphere(point3(0, 1, 0), 1.0, material1));

  auto material2 = new Lambertian(color(0.4, 0.2, 0.1));
  (*world)->add(new Sphere(point3(-4, 1, 0), 1.0, material2));

  auto material3 = new Metal(color(0.7, 0.6, 0.5), 0.0);
  (*world)->add(new Sphere(point3(4, 1, 0), 1.0, material3));

  (*world)->add(ray_model); 
  for(int i = 0; i<ray_model->meshes[0]->num_triangles; ++i) {
    ray_model->meshes[0]->triangles[i]->setMat(ground_material);
  }
  // (*world)->add(new Sphere(point3(0, 0, 3), 1, material3));
  // (*world)->add(new Triangle(point3(2,3,-1), point3(-5, 0, -1), point3(5, 0, -1), ground_material));
  // (*world)->add(new Sphere(point3(0, 0, -9), 3, ground_material));

}

int main() {
  // Metal test(color(3, 3, 3), 1.f);
  const float aspect_ratio = 16.0/9.0;
  RayRenderer renderer = RayRenderer(1000, aspect_ratio);

  point3 lookfrom(13,10,3);
  point3 lookat(0,0,0);
  vec3 vup(0,1,0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;

  // Allocate camera
  Camera* camera = new Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

  Model model = ModelLoader::load_model("models/cube/cube.obj");
  auto ray_cube = from_model(model);
  checkCudaErrors(cudaDeviceSynchronize());
    
  // Allocate world
  // Need to allocate a pointer so that we can create object with "new" in GPU so that virtual functions can run on device
  HittableList **world; 
  checkCudaErrors(cudaMallocManaged(&world, sizeof(HittableList*))); 
  checkCudaErrors(cudaDeviceSynchronize());
  random_scene<<<1, 1>>>(world, ray_cube);
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Done setting up scene\n");
  printf("Rendering...\n");
  clock_t start, stop;
  start = clock();

  color* fb = renderer.render_fb(camera, *world);

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "Render took " << timer_seconds << " seconds.\n";

  Window window("render", (unsigned int) renderer.get_image_width(), (unsigned int) renderer.get_image_height());


  WindowRenderer scene_renderer = WindowRenderer();
  unsigned int scene_texture = WindowRenderer::gen_scene_texture(fb, renderer.get_image_width(), renderer.get_image_height());

  while(!glfwWindowShouldClose(window.get_id())) {
    glClearColor(1.0f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scene_renderer.render_scene_to_window(scene_texture);
    
    glfwSwapBuffers(window.get_id());
    glfwPollEvents();
  }
}