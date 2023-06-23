#include "shader.h"

unsigned int Shader::compileShader(const char *shaderCode, GLenum shaderType) {
  int success;
  char infoLog[512];
  unsigned int shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, &shaderCode, NULL);
  glCompileShader(shader);

  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if(!success)
  {
    glGetShaderInfoLog(shader, 512, NULL, infoLog);
    std::cout << "Failed to compile shader" << infoLog << std::endl;
  };

  return shader;
}

Shader::Shader() {}
Shader::Shader(std::string vertexPath, std::string fragmentPath) {
  std::string vertexShaderString, fragmentShaderString;
  try {
    vertexShaderString = readFileToString(vertexPath);
    fragmentShaderString = readFileToString(fragmentPath);
  } catch (std::ifstream::failure e) {
    std::cout << "Shader file could not be read with path." << std::endl;
  }

  unsigned int vertexShader = compileShader(vertexShaderString.c_str(), GL_VERTEX_SHADER);
  unsigned int fragmentShader = compileShader(fragmentShaderString.c_str(), GL_FRAGMENT_SHADER);
  
  int success;
  char infoLog[512];
  id = glCreateProgram();
  glAttachShader(id, vertexShader);
  glAttachShader(id, fragmentShader);
  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if(!success) {
    glGetProgramInfoLog(id, 512, NULL, infoLog);
    std::cout << "Unable to link shader program" << infoLog << std::endl;
  }
  
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
}

void Shader::use() {
  glUseProgram(id);
}