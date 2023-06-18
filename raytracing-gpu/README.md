# Raytracer GPU

A GPU-based raytracer based on the following: [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
and [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- There were not much performance gains when parallelizing sampling (~8s for a randomly generated scene) 
- Sequentially sampling frees up memory to implement a "recursive" raytracing algorithm based on: [Iterative Layer-Based Raytracing in CUDA](https://www.eecis.udel.edu/~xli/publications/segovia2009iterative.pdf)