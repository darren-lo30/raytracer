#pragma once

#include "../math/vec3.h"
#include "shader.h"

class WindowRenderer {
  public:
    WindowRenderer();
    void render_scene_to_window(unsigned int scene_texture);
    static unsigned int gen_scene_texture(color *scene, int scene_width, int scene_height);
  private: 
    unsigned int VAO;
    Shader window_shader;
    static unsigned char *get_char_array_from_color_array(color *colors, int size);
};
