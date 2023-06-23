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
    Shader(std::string vertex_path, std::string fragmentPath);

    void use();
  private:
    static unsigned int compileShader(const char *shaderCode, GLenum shaderType);
};