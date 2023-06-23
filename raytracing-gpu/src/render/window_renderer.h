#pragma once

#include "../math/vec3.h"
#include "shader.h"

class WindowRenderer {
  public:
    WindowRenderer();
    void renderSceneToWindow(unsigned int sceneTexture);
    static unsigned int genSceneTexture(color *scene, int sceneWidth, int sceneHeight);
  private: 
    unsigned int VAO;
    Shader windowShader;
    static unsigned char *getCharArrayFromColorArray(color *colors, int size);
};
