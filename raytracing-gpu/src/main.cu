
#include <iostream>
#include <fstream>
#include "lib/utils.h"
#include "render/ray_renderer.h"
#include "lib/ray.h"
#include "lib/vec3.h"
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
#include <time.h>

__global__ void random_scene(HittableList** world) {
  if(threadIdx.x != 0 || blockIdx.x != 0) return;

  const int size = 25 * 25 + 10;
  Hittable** objects = new Hittable*[size];
  *world = new HittableList(size, objects);

  curandState state_val;
  curand_init(1984, 0, 0, &state_val);
  curandState *state = &state_val;



  auto ground_material = new Lambertian(color(0.5, 0.5, 0.5));
  (*world)->add(new Sphere(point3(0,-1000,0), 1000, ground_material));

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
}

int main() {
  // Metal test(color(3, 3, 3), 1.f);
  const float aspect_ratio = 16.0/9.0;
  RayRenderer renderer = RayRenderer(1200, aspect_ratio);

  point3 lookfrom(13,2,3);
  point3 lookat(0,0,0);
  vec3 vup(0,1,0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;

  // Allocate camera
  Camera* camera = new Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    
  // Allocate world
  // Need to allocate a pointer so that we can create object with "new" in GPU so that virtual functions can run on device
  HittableList **world; 
  checkCudaErrors(cudaMallocManaged(&world, sizeof(HittableList*))); 
  checkCudaErrors(cudaDeviceSynchronize());
  random_scene<<<1, 1>>>(world);
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Done setting up scene\n");

  std::ofstream myfile;
  myfile.open ("out/res.ppm", std::ofstream::out | std::ofstream::trunc);

  printf("Rendering...\n");
  clock_t start, stop;
  start = clock();

  color* fb = renderer.render_fb(camera, *world);

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "Render took " << timer_seconds << " seconds.\n";

  Window window("render", (unsigned int) renderer.get_image_width(), (unsigned int) renderer.get_image_height());
  Shader shader("shaders/vertex_shader.vs", "shaders/fragment_shader.fs");
  shader.use();
  float vertices[] = {
    // positions          // colors           // texture coords
     1.00f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 0.0f,  1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,  0.0f, 1.0f    // top left 
};

  unsigned int indices[] = {  // note that we start from 0!
      0, 1, 3,   // first triangle
      1, 2, 3    // second triangle
  };


  unsigned int VAO;
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(0);  
  glEnableVertexAttribArray(1);  

  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  unsigned char *data = get_char_array_from_color_array(fb, renderer.get_image_height() * renderer.get_image_width());
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1 );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, renderer.get_image_width(), renderer.get_image_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  while(!glfwWindowShouldClose(window.get_id())) {
    glClearColor(1.0f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader.use();
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window.get_id());
    glfwPollEvents();
  }

  myfile.close();
}