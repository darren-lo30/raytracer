
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
#include "materials/diffuse_light.h"
#include "utils/rand.h"
#include "objects/ray_model.h"

__global__ void randomScehen(HittableList** world, RayModel *rayModel) {
  if(threadIdx.x != 0 || blockIdx.x != 0) return;

  const int size = 25 * 25 + 20;
  Hittable** objects = new Hittable*[size];
  *world = new HittableList(size, objects);

  curandState stateVal;
  curand_init(1984, 0, 0, &stateVal);
  curandState *state = &stateVal;

  auto lightMat = new DiffuseLight(color(1, 0, 0));

  auto groundMaterial = new Lambertian(color(0.5, 0.5, 0.5));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto chooseMat = randomFloat(state);
      point3 center(a + 0.9*randomFloat(state), 0.2, b + 0.9*randomFloat(state));
      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        Material* sphereMat;

        if (chooseMat < 0.8) {
          // diffuse
          auto albedo = randomVec3(state) * randomVec3(state);
          sphereMat = new Lambertian(albedo);
        } else if (chooseMat < 0.95) {
          // metal
          auto albedo = randomVec3(state, 0.5, 1);
          auto fuzz = randomFloat(state, 0, 0.5);
          sphereMat = new Metal(albedo, fuzz);
        } else {
          // glass
          sphereMat = new Dielectric(1.5);
        }

        (*world)->add(new Sphere(center, 0.2, sphereMat));
      }
    }
  }

  auto material1 = new Dielectric(1.5);
  (*world)->add(new Sphere(point3(0, 5, 0), 0.5, lightMat));

  auto material2 = new Lambertian(color(0.4, 0.2, 0.1));
  (*world)->add(new Sphere(point3(-4, 1, 0), 1.0, material2));

  auto material3 = new Metal(color(0.7, 0.6, 0.5), 0.0);
  (*world)->add(new Sphere(point3(4, 1, 0), 1.0, material3));

  (*world)->add(rayModel); 
  for(int i = 0; i<rayModel->meshes[0]->numTriangles; ++i) {
    rayModel->meshes[0]->triangles[i]->setMat(lightMat);
  }
  // (*world)->add(new Sphere(point3(0, 0, 3), 1, material3));
  // (*world)->add(new Triangle(point3(2,3,-1), point3(-5, 0, -1), point3(5, 0, -1), groundMaterial));
  // (*world)->add(new Sphere(point3(0, 0, -9), 3, groundMaterial));
}

int main() {
  // Metal test(color(3, 3, 3), 1.f);
  const float aspectRatio = 16.0/9.0;
  RayRenderer renderer = RayRenderer(1000, aspectRatio);

  point3 lookfrom(-13,10,-3);
  point3 lookat(0,0,0);
  vec3 vup(0,1,0);
  auto distToFocus = 10.0;
  auto aperture = 0.1;

  // Allocate camera
  Camera* camera = new Camera(lookfrom, lookat, vup, 20, aspectRatio, aperture, distToFocus);

  Model model = ModelLoader::loadModel("models/cube/cube.obj");
  auto rayCube = fromModel(model);
  checkCudaErrors(cudaDeviceSynchronize());
    
  // Allocate world
  // Need to allocate a pointer so that we can create object with "new" in GPU so that virtual functions can run on device
  HittableList **world; 
  checkCudaErrors(cudaMallocManaged(&world, sizeof(HittableList*))); 
  checkCudaErrors(cudaDeviceSynchronize());
  randomScehen<<<1, 1>>>(world, rayCube);
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Done setting up scene\n");
  printf("Rendering...\n");
  clock_t start, stop;
  start = clock();

  color* fb = renderer.renderFb(camera, *world);

  stop = clock();
  double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "Render took " << timerSeconds << " seconds.\n";

  Window window("render", (unsigned int) renderer.getImageWidth(), (unsigned int) renderer.getImageHeight());


  WindowRenderer sceneRenderer = WindowRenderer();
  unsigned int sceneTexture = WindowRenderer::genSceneTexture(fb, renderer.getImageWidth(), renderer.getImageHeight());

  while(!glfwWindowShouldClose(window.getId())) {
    glClearColor(1.0f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    sceneRenderer.renderSceneToWindow(sceneTexture);
    
    glfwSwapBuffers(window.getId());
    glfwPollEvents();
  }
}