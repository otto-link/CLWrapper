project(clwrapper-macro-logger)

set(MACRO_LOGGER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/macro-logger/)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE ${MACRO_LOGGER_DIR})
