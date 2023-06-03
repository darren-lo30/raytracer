#pragma once

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "../utils/file.h"

class Shader {
  public:
    unsigned int id;

    Shader();
    Shader(std::string vertex_path, std::string fragment_path);

    void use();
  private:
    static unsigned int compile_shader(const char *shader_code, GLenum shader_type);
};