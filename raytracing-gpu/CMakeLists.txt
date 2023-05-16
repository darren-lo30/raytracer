cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)

project(raytracer-gpu VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE lib_src CONFIGURE_DEPENDS "src/*.cpp" "extern/src/glad.c")
add_library(library STATIC ${lib_src})
set_source_files_properties(${lib_src} PROPERTIES LANGUAGE CUDA)

target_compile_features(library PUBLIC cxx_std_17)
set_target_properties(library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(library PUBLIC extern/include)

find_package(glfw3 3.3 REQUIRED)


add_executable(raytracer-gpu src/main.cu)
set_property(TARGET raytracer-gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer-gpu PRIVATE library)
target_link_libraries(raytracer-gpu PRIVATE glfw)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CTest)
enable_testing()
