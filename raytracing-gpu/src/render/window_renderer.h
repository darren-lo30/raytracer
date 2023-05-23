#pragma once

#include "../lib/vec3.h"
#include "shader.h"

class WindowRenderer {
  static unsigned int gen_scene_texture(color *scene, int scene_width, int scene_height);
  static void render_scene_to_window(color *scene, int scene_width, int scene_height);
};
