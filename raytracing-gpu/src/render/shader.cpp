#include "shader.h"

unsigned int Shader::compile_shader(const char *shader_code, GLenum shader_type) {
  int success;
  char infoLog[512];
  unsigned int shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &shader_code, NULL);
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
Shader::Shader(std::string vertex_path, std::string fragment_path) {
  std::string vertex_shader_string, fragment_shader_string;
  try {
    vertex_shader_string = read_file_to_string(vertex_path);
    fragment_shader_string = read_file_to_string(fragment_path);
  } catch (std::ifstream::failure e) {
    std::cout << "Shader file could not be read with path." << std::endl;
  }

  unsigned int vertex_shader = compile_shader(vertex_shader_string.c_str(), GL_VERTEX_SHADER);
  unsigned int fragment_shader = compile_shader(fragment_shader_string.c_str(), GL_FRAGMENT_SHADER);
  
  int success;
  char infoLog[512];
  id = glCreateProgram();
  glAttachShader(id, vertex_shader);
  glAttachShader(id, fragment_shader);
  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if(!success) {
    glGetProgramInfoLog(id, 512, NULL, infoLog);
    std::cout << "Unable to link shader program" << infoLog << std::endl;
  }
  
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
}

void Shader::use() {
  glUseProgram(id);
}