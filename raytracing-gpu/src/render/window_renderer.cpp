#include "window_renderer.h"
#include "../utils/utils.h"

WindowRenderer::WindowRenderer() {
  // Init shader
  window_shader = Shader("shaders/vertex_shader.vs", "shaders/fragment_shader.fs");
  
  // Init VAO
  static float vertices[] = {
    1.00f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
    1.0f, -1.0f, 0.0f,  1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,  0.0f, 1.0f    // top left 
  };

  static unsigned int indices[] = {
    0, 1, 3,   // first triangle
    1, 2, 3    // second triangle
  };

  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  unsigned int VBO;
  glGenBuffers(1, &VBO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  
  unsigned int EBO;
  glGenBuffers(1, &EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(0);  
  glEnableVertexAttribArray(1);  
}

unsigned int WindowRenderer::gen_scene_texture(color *scene, int scene_width, int scene_height) {
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  unsigned char *data = get_char_array_from_color_array(scene, scene_height * scene_width);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1 );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scene_width, scene_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  return texture;
}

void WindowRenderer::render_scene_to_window(unsigned int scene_texture) {
  window_shader.use();
  glBindTexture(GL_TEXTURE_2D, scene_texture);
  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}


unsigned char *WindowRenderer::get_char_array_from_color_array(color *colors, int size) {
  unsigned char *char_colors = new unsigned char[size * 3];
  for(int i = 0; i<size; ++i) {
    char_colors[i * 3 + 0] = static_cast<unsigned char>(colors[i].x() * 255);
    char_colors[i * 3 + 1] = static_cast<unsigned char>(colors[i].y() * 255);;
    char_colors[i * 3 + 2] = static_cast<unsigned char>(colors[i].z() * 255);;
  }

  return char_colors;
}