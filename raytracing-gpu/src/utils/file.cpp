#include "file.h"

std::string readFileToString(std::string path) {
  std::fstream file;

  file.open(path);

  std::stringstream fileStringStream;
  fileStringStream << file.rdbuf();

  file.close();

  return fileStringStream.str();
}