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
struct RenderHitRecord {
  color hit_color;
  ray out_ray;
  bool is_first_layer;
};

struct RenderHitLayer {
  color hit_color;
};

__device__
float to_gamma_space(float val) {
  return sqrt(clamp(val, 0.0, 0.999));
}


__global__
void in_rays_init(ray *in_rays, int image_width, int image_height, const Camera *camera, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;

  curandState* local_state = &rand_states[pixel_idx];
  float h = float(x + random_float(local_state)) / image_width;
  float v = float(y + random_float(local_state)) / image_height;
  in_rays[pixel_idx] = camera->get_ray(h, v, local_state);
}

__global__ void first_layer_num_init(char *first_layer_num, int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;

  first_layer_num[pixel_idx] = -1;
}

__device__ 
RenderHitRecord ray_color_step(const ray& in_ray, const HittableList *world, curandState* rand_state) {    
  // Implies that the in_ray is null
  if(in_ray.is_null_ray()) return { color(0, 0, 0), ray::null_ray()};
  
  // const color env_color = color(0.3, 0.3, 0.3);
  HitRecord rec;    
  if (!world->hit(in_ray, 0.001, infinity, rec)) {
    vec3 unit_direction = unit(in_ray.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    color env_color = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);

    return { env_color, ray::null_ray(), true };
  } else {
    color attenuation;
    ray scattered;
    if(rec.mat->scatter(in_ray, rec, attenuation, scattered, rand_state)) {
      return { attenuation, scattered, false };
    } else {
      return { color(0, 0, 0), ray::null_ray(), false };
    }
  }
}

__global__
void ray_trace_step(ray *in_rays, RenderHitLayer *render_layer, char *first_layer_num, int layer_num, int image_width, int image_height, const Camera *camera, const HittableList* world,  curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  
  RenderHitRecord hit_record = ray_color_step(in_rays[pixel_idx], world, &rand_states[pixel_idx]);
  render_layer[pixel_idx] = { hit_record.hit_color };
  in_rays[pixel_idx] = hit_record.out_ray;
  if(hit_record.is_first_layer) {
    first_layer_num[pixel_idx] = layer_num; 
  }
}

__global__
void rand_state_init(int image_width, int image_height, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;

  curand_init(1984, pixel_idx, 0, &rand_states[pixel_idx]);
}

__global__ void combine_layers(color *prev_layers, RenderHitLayer *render_layer, char *first_layer_num, int layer_num,  int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  if(first_layer_num[pixel_idx] != -1) {
    if(first_layer_num[pixel_idx] == layer_num) {
      prev_layers[pixel_idx] = render_layer[pixel_idx].hit_color;
    } else if(first_layer_num[pixel_idx] > layer_num) {
      prev_layers[pixel_idx] = prev_layers[pixel_idx] * render_layer[pixel_idx].hit_color;
    }
  }
}

__global__
void accumulate_sample(color *fb, int num_samples, color *fb_sample, int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  fb[pixel_idx] += fb_sample[pixel_idx] / num_samples;
}

__global__
void fb_to_gamma_space(color *fb, int image_width, int image_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= image_width || y >= image_height) return;
  int pixel_idx = y * image_width + x;
  color &c = fb[pixel_idx];
  fb[pixel_idx] = color(to_gamma_space(c.x()), to_gamma_space(c.y()), to_gamma_space(c.z()));
}

class RayRenderer {
  public:
    RayRenderer(int image_width, int image_height) : image_width(image_width), image_height(image_height) {  
      int num_pixels = image_width * image_height;
      auto fb_size = num_pixels * sizeof(color);

      checkCudaErrors(cudaMallocManaged(&fb, fb_size));
      checkCudaErrors(cudaMallocManaged(&fb_sample, fb_size));
      checkCudaErrors(cudaMallocManaged(&first_layer_num, num_pixels * sizeof(char)));
      checkCudaErrors(cudaMallocManaged(&in_rays, num_pixels * sizeof(ray)));
      checkCudaErrors(cudaMallocManaged(&render_layer, num_pixels * sizeof(RenderHitLayer) * max_depth));
      checkCudaErrors(cudaMallocManaged(&rand_states, sizeof(curandState) * num_pixels));
      checkCudaErrors(cudaDeviceSynchronize());
    }

    RayRenderer(int image_width, float aspect_ratio) : RayRenderer(image_width, static_cast<int>(image_width / aspect_ratio)) {}

    int get_image_width() const { return image_width; };
    int get_image_height() const { return image_height; };

    color* render_fb(const Camera *camera, const HittableList *world) const {
      int num_pixels = image_width * image_height;

      // 8 x 8 grid of threads in a block
      int thread_dim = 8;
      dim3 threads(thread_dim, thread_dim);
      dim3 blocks((image_width + thread_dim - 1)/thread_dim, (image_height + thread_dim - 1)/thread_dim);

      rand_state_init<<<blocks, threads>>>(image_width, image_height, rand_states);
      checkCudaErrors(cudaDeviceSynchronize()); 


      for(int sample_num = 0; sample_num<num_samples; ++sample_num) {
        first_layer_num_init<<<blocks, threads>>>(first_layer_num, image_width, image_height);
        checkCudaErrors(cudaDeviceSynchronize()); 

        in_rays_init<<<blocks, threads>>>(in_rays, image_width, image_height, camera, rand_states);
        checkCudaErrors(cudaDeviceSynchronize()); 

        for(int layer_num = 0; layer_num<max_depth; ++layer_num) {
          ray_trace_step<<<blocks, threads>>>(in_rays, &render_layer[layer_num * num_pixels], first_layer_num, layer_num, image_width, image_height, camera, world, rand_states);
          checkCudaErrors(cudaDeviceSynchronize());
        }

        for(int layer_num = max_depth - 1; layer_num >= 0; --layer_num) {
          combine_layers<<<blocks, threads>>>(fb_sample, &render_layer[layer_num * num_pixels], first_layer_num, layer_num, image_width, image_height);
          checkCudaErrors(cudaDeviceSynchronize());
        }

        accumulate_sample<<<blocks, threads>>>(fb, num_samples, fb_sample, image_width, image_height);
        checkCudaErrors(cudaDeviceSynchronize());
      }

      fb_to_gamma_space<<<blocks, threads>>>(fb, image_width, image_height);
      checkCudaErrors(cudaDeviceSynchronize()); 

      return fb;
    }
  private:
    color *fb;
    color *fb_sample;
    ray *in_rays;
    char *first_layer_num;
    RenderHitLayer *render_layer;
    curandState *rand_states;

    int image_width;
    int image_height;    
    const int max_depth = 50;
    const int num_samples = 10;
};

#endif