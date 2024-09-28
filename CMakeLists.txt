cmake_minimum_required(VERSION 3.20)

project(pro LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 路径
set(CUDA_PATH "/usr/local/cuda-11.8")

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 86) 

# 启用 CUDA 分离编译
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 生成 compile_commands.json
# add_compile_options(-Wno-deprecated-declarations)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUTLASS 头文件
set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include)

set(CUDA_HEADER_DIRS 
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include
    ${CUTLASS_HEADER_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_LIB_DIRS 
    ${CUDA_PATH}/lib64)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fPIC --expt-extended-lambda")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true -rdc=true")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# list(APPEND COMMON_LIB_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

include_directories(${CUDA_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)

# add_library(transformer-shared SHARED
#   # $<TARGET_OBJECTS:activation_kernels>
#   $<TARGET_OBJECTS:logger>
#   $<TARGET_OBJECTS:memory_utils>
# )
 
# add_subdirectory(3rdparty)
add_subdirectory(src)
# add_subdirectory(examples)

# 添加 CUDA 编译选项
# target_compile_options(demo_vit PUBLIC
#     $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
# )

