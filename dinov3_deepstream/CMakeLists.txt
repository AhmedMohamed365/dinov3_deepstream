cmake_minimum_required(VERSION 3.16)
project(dinov3_deepstream LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-rtsp-server-1.0)

find_package(CUDAToolkit REQUIRED)

# DeepStream root (adjust if needed)
set(DEEPSTREAM_ROOT /opt/nvidia/deepstream/deepstream-8.0)

add_library(nvdsinfer_custom_detection SHARED
    src/custom_parsers/nvdsinfer_custom_detection.cpp
)

add_executable(dinov3_deepstream
    src/main.cpp
    src/probes/dinov3_probe.cpp
    src/probes/depth_probe.cpp
    src/probes/segmentation_probe.cpp
    src/probes/optical_flow_probe.cpp
    src/pipeline/pipeline_builder.cpp
    src/utils_cuda/depth.cu
    src/utils_cuda/depth_percentile.cu
    src/utils_cuda/segmentation.cu
    src/utils_cuda/optical_flow.cu
)

set_target_properties(nvdsinfer_custom_detection PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Let CMake compile CUDA properly
set_target_properties(dinov3_deepstream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(nvdsinfer_custom_detection PRIVATE
    ${DEEPSTREAM_ROOT}/sources/includes
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(dinov3_deepstream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # Allow #include "config/app_config.h"
    ${GST_INCLUDE_DIRS}
    ${DEEPSTREAM_ROOT}/sources/includes
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(nvdsinfer_custom_detection PRIVATE
    ${DEEPSTREAM_ROOT}/lib
)

target_link_directories(dinov3_deepstream PRIVATE
    ${GST_LIBRARY_DIRS}
    ${DEEPSTREAM_ROOT}/lib
)

target_compile_options(dinov3_deepstream PRIVATE
    ${GST_CFLAGS_OTHER}
)

target_link_libraries(nvdsinfer_custom_detection PRIVATE
    nvds_inferutils
    CUDA::cudart
)

# Minimal DS libs for metadata access + gstreamer libs
target_link_libraries(dinov3_deepstream PRIVATE
    ${GST_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    CUDA::cudart
)