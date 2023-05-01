#include <iostream>
#include <fstream>
#include "lib/utils.h"
#include "lib/renderer.h"
#include "lib/color.h"
#include "lib/ray.h"
#include "lib/vec3.h"
#include "lib/camera.h"
#include "lib/hittable.h"
#include "lib/hittable_list.h"
#include "lib/sphere.h"
#include "lib/metal.h"
#include "lib/lambertian.h"
#include "lib/dielectric.h"

using namespace std;


color ray_color(const ray& r, const Hittable& world, int depth) {
  if (depth <= 0) return color(0,0,0);

  HitRecord rec;    
  if (world.hit(r, 0.001, infinity, rec)) {
    color attenuation;
    ray scattered;
    if(rec.mat->scatter(r, rec, attenuation, scattered)) {
      return attenuation * ray_color(scattered, world, depth - 1);
    }
    
    return color(0, 0, 0);
  }
  vec3 unit_direction = unit(r.direction());
  auto t = 0.5*(unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}


HittableList generate_random_scene() {
  HittableList world;

  auto ground_material = make_shared<Lambertian>(color(0.5, 0.5, 0.5));
  world.add(make_shared<Sphere>(point3(0,-1000,0), 1000, ground_material));

  for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<Lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<Sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<Metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<Sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

int main() {
  const double aspect_ratio = 16.0/9.0;
  Renderer renderer = Renderer(400, aspect_ratio);
  HittableList world = generate_random_scene();


  point3 lookfrom(13,2,3);
  point3 lookat(0,0,0);
  vec3 vup(0,1,0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;

  Camera camera(lookfrom, lookat, vup, 20, 16.0/9.0, aperture, dist_to_focus);
  
  const int num_samples = 100;
  const int max_depth = 50;

  ofstream myfile;
  myfile.open ("out/res.ppm", ofstream::out | ofstream::trunc);

  myfile << "P3\n" << renderer.image_width << " " << renderer.image_height << " 255\n";

  for(int i = renderer.image_height-1; i>=0; --i) {
    std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
    for(int j = 0; j<renderer.image_width; ++j) {
      color col = color(0, 0, 0);
      for(int k = 0; k<num_samples; ++k) {
        double h = (j + random_double()) / (renderer.image_width - 1); // Between 0 and 1
        double v = (i + random_double()) / (renderer.image_height - 1);
  
        ray r = camera.get_ray(h, v);
        col += ray_color(r, world, max_depth);
      }

      draw_color(myfile, col, num_samples);
    }
  }

  myfile.close();
  

}