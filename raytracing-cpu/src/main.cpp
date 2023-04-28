#include <iostream>
#include <fstream>
#include "lib/renderer.h"
#include "lib/color.h"
#include "lib/ray.h"
#include "lib/vec3.h"
#include "lib/camera.h"

using namespace std;


bool hit_sphere(const point3 &center, double radius, const ray &r) {
  vec3 oc = r.origin() - center;
  double a = dot(r.direction(), r.direction());
  double b = 2.0 * dot(r.direction(), oc);
  double c = dot(oc, oc) - radius * radius;
  double determinant = b*b - 4*a*c;

  return determinant > 0; 
}

color ray_color(const ray &r) {
  if(hit_sphere(point3(0, 0, -1), 0.5, r)) {
    return color(1, 0, 0);
  }

  vec3 unit_direction = unit(r.direction()); 
  double t = 0.5*(unit_direction.y() + 1.0); // Scale between 0.5 and 1
  cout << t << endl;
  return t * color(1, 1, 1) + (1-t) * color(0.5, 0.7, 1.0);
}
int main() {
  Renderer renderer = Renderer(256, 16.0/9.0);
  Camera camera = Camera(renderer.aspect_ratio() * 2.0, 2.0, 1.0);

  ofstream myfile;
  myfile.open ("out/res.ppm", ofstream::out | ofstream::trunc);

  myfile << "P3\n" << renderer.image_width << " " << renderer.image_height << " 255\n";

  for(int i = 0; i<renderer.image_height; ++i) {
    for(int j = 0; j<renderer.image_width; ++j) {
      double h = double(j) / (renderer.image_width - 1); // Between 0 and 1
      double v = double(i) / (renderer.image_height - 1);
      
      ray r = ray(camera.position, camera.lower_left_corner() + h * camera.horizontal()  + v * camera.vertical() - camera.position);
      draw_color(myfile, ray_color(r));
    }
  }

  myfile.close();
  

}