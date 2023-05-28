cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)

project(raytracer-gpu VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE lib_src CONFIGURE_DEPENDS "src/*.cpp" "extern/glad/src/glad.c")
add_library(library STATIC ${lib_src})
set_source_files_properties(${lib_src} PROPERTIES LANGUAGE CUDA)

target_compile_features(library PUBLIC cxx_std_17)
set_target_properties(library PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(library PUBLIC extern/glad/include extern/assimp-src/include extern/assimp-build/include)

find_package(glfw3 3.3 REQUIRED)

# Asssimp
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/extern CACHE PATH "Missing description." FORCE)
FetchContent_Declare(assimp
GIT_REPOSITORY https://github.com/assimp/assimp.git
GIT_TAG master)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

# Compile program
add_executable(raytracer-gpu src/main.cu)
set_property(TARGET raytracer-gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer-gpu library glfw assimp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CTest)
enable_testing()
