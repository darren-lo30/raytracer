#ifndef RENDERER_H
#define RENDERER_H

class Renderer {
  public:
    Renderer(int image_width, int image_height) : image_width(image_width), image_height(image_height) {}
    Renderer(int image_width, double aspect_ratio) : image_width(image_width)
    {
      image_height = static_cast<int>(image_width / aspect_ratio);
    }

    double aspect_ratio() {
      return (double) image_width / image_height;
    }
    int image_width;
    int image_height;    
};

#endif