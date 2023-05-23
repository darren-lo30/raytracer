#include "file.h"

std::string read_file_to_string(std::string path) {
  std::fstream file;

  file.open(path);

  std::stringstream file_string_stream;
  file_string_stream << file.rdbuf();

  file.close();

  return file_string_stream.str();
}