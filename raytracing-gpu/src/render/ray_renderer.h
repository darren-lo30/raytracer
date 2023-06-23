#pragma once

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
  color hitColor;
  color emitColor;
  ray outRay;
  bool isFirstLayer;
};

struct RenderHitLayer {
  color hitColor;
  color emitColor;
};

__device__
float toGammaSpace(float val) {
  return sqrt(clamp(val, 0.0, 0.999));
}

__global__
void fbRenderLayerInit(ray *inRays, char *firstLayerNum, int imageWidth, int imageHeight, const Camera *camera, curandState *randStates) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;

  curandState* localState = &randStates[pixelIdx];
  float h = float(x + randomFloat(localState)) / imageWidth;
  float v = float(y + randomFloat(localState)) / imageHeight;
  inRays[pixelIdx] = camera->getRay(h, v, localState);
  
  firstLayerNum[pixelIdx] = -1;
}

__device__ 
RenderHitRecord rayColorStep(const ray& inRay, const HittableList *world, curandState* randState) {    
  // Implies that the inRay is null
  if(inRay.isNullRay()) return { color(0, 0, 0), color(0, 0, 0), ray::nullRay(), false};
  
  const color envColor = color(0, 0, 0);
  HitRecord rec;    
  if (!world->hit(inRay, 0.001, infinity, rec)) {
    return { envColor, color(0, 0, 0), ray::nullRay(), true };
  } else {
    color attenuation;
    ray scattered;
    color emitted = rec.mat->emit(rec);
    if(rec.mat->scatter(inRay, rec, attenuation, scattered, randState)) {
      return { attenuation, emitted, scattered, false };
    } else {
      return { color(0, 0, 0), emitted, ray::nullRay(), true };
    }
  }
}

__global__
void rayTraceStep(ray *inRays, RenderHitLayer *renderLayer, char *firstLayerNum, int layerNum, int imageWidth, int imageHeight, const Camera *camera, const HittableList* world,  curandState *randStates) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;
  
  RenderHitRecord hitRecord = rayColorStep(inRays[pixelIdx], world, &randStates[pixelIdx]);
  renderLayer[pixelIdx] = { hitRecord.hitColor, hitRecord.emitColor };
  inRays[pixelIdx] = hitRecord.outRay;
  if(hitRecord.isFirstLayer) {
    firstLayerNum[pixelIdx] = layerNum; 
  }
}

__global__
void randStateInit(int imageWidth, int imageHeight, curandState *randStates) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;

  curand_init(1984, pixelIdx, 0, &randStates[pixelIdx]);
}

__global__ void combineLayers(color *prevLayers, RenderHitLayer *renderLayer, char *firstLayerNum, int layerNum,  int imageWidth, int imageHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;
  if(firstLayerNum[pixelIdx] != -1) {
    if(firstLayerNum[pixelIdx] == layerNum) {
      prevLayers[pixelIdx] = renderLayer[pixelIdx].hitColor + renderLayer[pixelIdx].emitColor;
    } else if(firstLayerNum[pixelIdx] > layerNum) {
      prevLayers[pixelIdx] = prevLayers[pixelIdx] * renderLayer[pixelIdx].hitColor + renderLayer[pixelIdx].emitColor;
    }
  }
}

__global__
void accumulateSamples(color *fb, int numSamples, color *fbSample, int imageWidth, int imageHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;
  fb[pixelIdx] += fbSample[pixelIdx] / numSamples;
}

__global__
void fbToGammaSpace(color *fb, int imageWidth, int imageHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imageWidth || y >= imageHeight) return;
  int pixelIdx = y * imageWidth + x;
  color &c = fb[pixelIdx];
  fb[pixelIdx] = color(toGammaSpace(c.x()), toGammaSpace(c.y()), toGammaSpace(c.z()));
}

class RayRenderer {
  public:
    RayRenderer(int imageWidth, int imageHeight) : imageWidth(imageWidth), imageHeight(imageHeight) {  
      int numPixels = imageWidth * imageHeight;
      auto fbSize = numPixels * sizeof(color);

      checkCudaErrors(cudaMallocManaged(&fb, fbSize));
      checkCudaErrors(cudaMallocManaged(&fbSample, fbSize));
      checkCudaErrors(cudaMallocManaged(&firstLayerNum, numPixels * sizeof(char)));
      checkCudaErrors(cudaMallocManaged(&inRays, numPixels * sizeof(ray)));
      checkCudaErrors(cudaMallocManaged(&renderLayer, numPixels * sizeof(RenderHitLayer) * maxDepth));
      checkCudaErrors(cudaMallocManaged(&randStates, sizeof(curandState) * numPixels));
      checkCudaErrors(cudaDeviceSynchronize());
    }

    RayRenderer(int imageWidth, float aspectRation) : RayRenderer(imageWidth, static_cast<int>(imageWidth / aspectRation)) {}

    int getImageWidth() const { return imageWidth; };
    int getImageHeight() const { return imageHeight; };

    void renderFbSamples(int sampleNum, const Camera *camera, const HittableList *world) const {
      int numPixels = imageWidth * imageHeight;

      int threadDim = 8;
      dim3 threads(threadDim, threadDim);
      dim3 blocks((imageWidth + threadDim - 1)/threadDim, (imageHeight + threadDim - 1)/threadDim);

      fbRenderLayerInit<<<blocks, threads>>>(inRays, firstLayerNum, imageWidth, imageHeight, camera, randStates);
      checkCudaErrors(cudaDeviceSynchronize()); 

      for(int layerNum = 0; layerNum<maxDepth; ++layerNum) {
        rayTraceStep<<<blocks, threads>>>(inRays, &renderLayer[layerNum * numPixels], firstLayerNum, layerNum, imageWidth, imageHeight, camera, world, randStates);
        checkCudaErrors(cudaDeviceSynchronize());
      }

      for(int layerNum = maxDepth - 1; layerNum >= 0; --layerNum) {
        combineLayers<<<blocks, threads>>>(fbSample, &renderLayer[layerNum * numPixels], firstLayerNum, layerNum, imageWidth, imageHeight);
        checkCudaErrors(cudaDeviceSynchronize());
      }

      accumulateSamples<<<blocks, threads>>>(fb, numSamples, fbSample, imageWidth, imageHeight);
      checkCudaErrors(cudaDeviceSynchronize());
    }

    color* renderFb(const Camera *camera, const HittableList *world) const {
      int threadDim = 8;
      dim3 threads(threadDim, threadDim);
      dim3 blocks((imageWidth + threadDim - 1)/threadDim, (imageHeight + threadDim - 1)/threadDim);

      randStateInit<<<blocks, threads>>>(imageWidth, imageHeight, randStates);
      checkCudaErrors(cudaDeviceSynchronize()); 

      for(int sampleNum = 0; sampleNum<numSamples; ++sampleNum) {
        renderFbSamples(sampleNum, camera, world);
      }

      fbToGammaSpace<<<blocks, threads>>>(fb, imageWidth, imageHeight);
      checkCudaErrors(cudaDeviceSynchronize()); 

      return fb;
    }
  private:
    color *fb;
    color *fbSample;
    ray *inRays;
    char *firstLayerNum;
    RenderHitLayer *renderLayer;
    curandState *randStates;

    int imageWidth;
    int imageHeight;    
    const int maxDepth = 10;
    const int numSamples = 10;
};