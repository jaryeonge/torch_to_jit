cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

file(STRINGS "version.txt" APP_VERSION)
message(STATUS "VERSION: ${APP_VERSION}")
project(torchtest VERSION ${APP_VERSION})

find_package(Torch REQUIRED)
add_executable(torchtest-app torchtest-app.cpp)
target_link_libraries(torchtest-app "${TORCH_LIBRARIES}")

set_property(TARGET torchtest-app PROPERTY CXX_STANDARD 14)