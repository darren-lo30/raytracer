cmake_minimum_required(VERSION 3.8)
project(raytracer-gpu VERSION 0.1.0 LANGUAGES CUDA CXX)

enable_language(CUDA)


include(CTest)
enable_testing()

add_executable(raytracer-gpu src/main.cu)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
