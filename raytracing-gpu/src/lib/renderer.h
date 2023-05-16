#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "hittable_list.h"
#include "vec3.h"
#include "../materials/material.h"
#include "error.h"
#include "utils.h"
#include <fstream>
#include "hittable.h"

float to_gamma_space(float val) {
  return sqrt(val);
}

void draw_color(std::ostream &out, color pixel_color) {
  float r = clamp(to_gamma_space(pixel_color.x()), 0, 0.999); 
  float g = clamp(to_gamma_space(pixel_color.y()), 0, 0.999); 
  float b = clamp(to_gamma_space(pixel_color.z()), 0, 0.999); 

  int rVal = static_cast<int>(r * 256);
  int gVal = static_cast<int>(g * 256);
  int bVal = static_cast<int>(b * 256);

  out << rVal << ' ' << gVal << ' ' << bVal << '\n';
}

__device__ 
color ray_color(const ray& in_ray, const HittableList *world, curandState* rand_state) {    
  const int max_depth = 50;

  ray curr_ray = in_ray;
  color out_col(1, 1, 1);

  for(int i = 0; i<max_depth; ++i) {
    HitRecord rec;    
    if (world->hit(curr_ray, 0.001, infinity, rec)) {
      color attenuation;
      ray scattered;
      if(rec.mat->scatter(curr_ray, rec, attenuation , scattered, rand_state)) {
        out_col = out_col * attenuation;
        curr_ray = scattered;
      }
    } else {
      vec3 unit_direction = unit(curr_ray.direction());
      auto t = 0.5*(unit_direction.y() + 1.0);
      return out_col * ((1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));  
    }
  }

  return color(0, 0, 0);
}

__global__
void render_fb_samples(color *fb_samples, int image_width, int image_height, const Camera *camera, const HittableList* world, curandState *rand_states) {
  int x = blockIdx.x;
  int y = blockIdx.y;

  if(x >= image_width || y >= image_height) return;

  int pixel_idx = y * image_width + x;
  int sample_idx = pixel_idx * blockDim.x + threadIdx.x;

  curandState* local_state = &rand_states[sample_idx];
  float h = float(x + random_float(local_state)) / image_width;
  float v = float(y + random_float(local_state)) / image_height;
  fb_samples[sample_idx] = ray_color(camera->get_ray(h, v, &rand_states[sample_idx]), world, local_state);
}

__global__
void average_samples(color *fb_samples, int num_samples, color *fb, int image_width, int image_height) {
  int x = blockIdx.x;
  int y = blockIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  for(int i = 0; i<num_samples; ++i) {
    fb[pixel_idx] += fb_samples[pixel_idx * num_samples + i];
  }

  fb[pixel_idx] /= num_samples;
}

__global__
void rand_state_init(int image_width, int image_height, curandState *rand_states) {
  int x = blockIdx.x;
  int y = blockIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  int sample_idx = pixel_idx * blockDim.x + threadIdx.x;

  curand_init(1984, sample_idx, 0, &rand_states[sample_idx]);
}

class Renderer {
  public:
    Renderer(int image_width, int image_height) : image_width(image_width), image_height(image_height) {}
    Renderer(int image_width, float aspect_ratio) : image_width(image_width)
    {
      image_height = static_cast<int>(image_width / aspect_ratio);
    }

    color* render_fb(const Camera *camera, const HittableList *world) const {
      int num_pixels = image_width * image_height;
      auto fb_size = num_pixels * sizeof(color);
      auto fb_samples_size = fb_size * num_samples;

      color *fb;
      color *fb_samples;
      curandState *rand_states;

      checkCudaErrors(cudaMallocManaged(&fb, fb_size));
      checkCudaErrors(cudaMallocManaged(&fb_samples, fb_samples_size));
      checkCudaErrors(cudaMalloc(&rand_states, sizeof(curandState) * num_pixels * num_samples));
      checkCudaErrors(cudaDeviceSynchronize());

      dim3 blocks(image_width, image_height);

      rand_state_init<<<blocks, num_samples>>>(image_width, image_height, rand_states);
      checkCudaErrors(cudaDeviceSynchronize());

      render_fb_samples<<<blocks, num_samples>>>(fb_samples, image_width, image_height, camera, world, rand_states);
      checkCudaErrors(cudaDeviceSynchronize());

      average_samples<<<blocks, 1>>>(fb_samples, num_samples, fb, image_width, image_height);
      checkCudaErrors(cudaDeviceSynchronize());

      return fb;
    }
    
    void render_out(const color* fb, std::ostream& out) {
      out << "P3\n" << image_width << " " << image_height << " 255\n";

      for(int i = image_height-1; i>=0; --i) {
      std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
      for(int j = 0; j<image_width; ++j) {
        int pixel_idx = i * image_width + j;
        color out_col = fb[pixel_idx];
        draw_color(out, out_col);
      }
    }
  }

  private:
    int image_width;
    int image_height;    
    const int max_depth = 50;
    const int num_samples = 10;
};

#endif