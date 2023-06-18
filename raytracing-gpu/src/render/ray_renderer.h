#ifndef RayRenderer_H
#define RayRenderer_H

#include "camera.h"
#include "../objects/hittable_list.h"
#include "../math/vec3.h"
#include "../materials/material.h"
#include "error.h"
#include <fstream>
#include "../objects/hittable.h"
#include "../utils/constants.h"
#include "../utils/rand.h"

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

// __device__ 
// color ray_color(const ray& in_ray, const HittableList *world, const int max_depth, curandState* rand_state) {    
//   const color env_color = ((0.7 * color(1, 1, 1)) + (0.3 * color(0.5, 0.7, 1.0)));
//   // const color env_color = color(0, 0, 0);

//   ray curr_ray = in_ray;
//   color out_col(1, 1, 1);

//   for(int i = 0; i<max_depth; ++i) {
//     HitRecord rec;    
//     if (world->hit(curr_ray, 0.001, infinity, rec)) {
//       color attenuation;
//       ray scattered;
//       // color emitted = rec.mat->emit(rec);
//       if(rec.mat->scatter(curr_ray, rec, attenuation, scattered, rand_state)) {
//         out_col = 0.5 * attenuation + 0.5 * out_col;
//         // out_col = out_col * attenuation;
//         curr_ray = scattered;
//       }
//     } else {
//       vec3 unit_direction = unit(curr_ray.direction());
//       return out_col * env_color;  
//     }
//   }

//   return color(0, 0, 0);
// }

__device__ 
color ray_color(const ray& in_ray, const HittableList *world, const int max_depth, curandState* rand_state) {    
  const color env_color = (0.7 * color(1, 1, 1)) + (0.3 * color(0.5, 0.7, 1.0));
  HitRecord rec;    
  if (max_depth <= 0 || !world->hit(in_ray, 0.001, infinity, rec)) {
    return env_color;
  } else {
    color attenuation;
    ray scattered;
    // color emitted = rec.mat->emit(rec);
    if(rec.mat->scatter(in_ray, rec, attenuation, scattered, rand_state)) {
      // out_col = 0.5 * attenuation + 0.5 * out_col;
      return 0.5 * attenuation + 0.5 * ray_color(scattered, world, max_depth - 1, rand_state);
    } else {
      return color(0, 0, 0);
    }
  }
}

__global__
void render_fb_sample(color *fb_sample, int image_width, int image_height, const Camera *camera, const HittableList* world, const int max_depth, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;

  curandState* local_state = &rand_states[pixel_idx];
  float h = float(x + random_float(local_state)) / image_width;
  float v = float(y + random_float(local_state)) / image_height;
  fb_sample[pixel_idx] = ray_color(camera->get_ray(h, v, local_state), world, max_depth, local_state);
}

__global__
void accumulate_sample(color *fb_sample, int num_samples, color *fb, int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  fb[pixel_idx] += fb_sample[pixel_idx] / num_samples;
}

__global__
void rand_state_init(int image_width, int image_height, curandState *rand_states, int sample_num) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;

  curand_init(1984, pixel_idx * sample_num, 0, &rand_states[pixel_idx]);
}

class RayRenderer {
  public:
    RayRenderer(int image_width, int image_height) : image_width(image_width), image_height(image_height) {  
      int num_pixels = image_width * image_height;
      auto fb_size = num_pixels * sizeof(color);
      auto fb_sample_size = fb_size;

      checkCudaErrors(cudaMallocManaged(&fb, fb_size));
      checkCudaErrors(cudaMallocManaged(&fb_sample, fb_sample_size));
      checkCudaErrors(cudaMallocManaged(&rand_states, sizeof(curandState) * num_pixels));
      checkCudaErrors(cudaDeviceSynchronize());
    }

    RayRenderer(int image_width, float aspect_ratio) : image_width(image_width)
    {
      RayRenderer(image_width, static_cast<int>(image_width / aspect_ratio));
    }

    int get_image_width() const { return image_width; };
    int get_image_height() const { return image_height; };

    color* render_fb(const Camera *camera, const HittableList *world) const {
      // 8 x 8 grid of threads in a block
      int thread_dim = 8;
      dim3 threads(thread_dim, thread_dim);
      dim3 blocks((image_width + thread_dim - 1)/thread_dim, (image_height + thread_dim - 1)/thread_dim);

      for(int sample_num = 0; sample_num<num_samples; ++sample_num) {
        rand_state_init<<<blocks, threads>>>(image_width, image_height, rand_states, sample_num);
        checkCudaErrors(cudaDeviceSynchronize()); 

        render_fb_sample<<<blocks, threads>>>(fb_sample, image_width, image_height, camera, world, max_depth, rand_states);
        checkCudaErrors(cudaDeviceSynchronize());
        
        accumulate_sample<<<blocks, threads>>>(fb_sample, num_samples, fb, image_width, image_height);
        checkCudaErrors(cudaDeviceSynchronize());
      }

      return fb;
    }
  private:
    color *fb;
    color *fb_sample;
    curandState *rand_states;

    int image_width;
    int image_height;    
    const int max_depth = 4;
    const int num_samples = 50;
};

#endif