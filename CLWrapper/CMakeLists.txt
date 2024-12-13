cmake_minimum_required(VERSION 3.22)

project(clwrapper)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_link_libraries(${PROJECT_NAME} clerrorlookup ${OpenCL_LIBRARIES}
                      clwrapper-macro-logger)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
