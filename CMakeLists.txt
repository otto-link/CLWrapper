cmake_minimum_required(VERSION 3.22)

project(clwrapper-root)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package(spdlog REQUIRED)
find_package(OpenCL REQUIRED)
add_definitions("-DCL_HPP_MINIMUM_OPENCL_VERSION=120")
add_definitions("-DCL_HPP_TARGET_OPENCL_VERSION=120")

add_subdirectory(${PROJECT_SOURCE_DIR}/CLWrapper)
add_subdirectory(${PROJECT_SOURCE_DIR}/external)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
