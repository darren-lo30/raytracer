# raytracer
A ray tracer made in C++. I first built a ray tracer implemented to only use the CPU. Afterwards, I extended this implementation to use the GPU with CUDA. The algorithm used is recursive in order to achieve more realistic lighting. 

I follow this paper proposing a way to effectively do this on the GPU:
https://www.eecis.udel.edu/~xli/publications/segovia2009iterative.pdf 

Rather than writing a recursive function in CUDA, it iteratively generates each layer placing them onto a stack and combines them at the end achieving the same effect as recursion.
