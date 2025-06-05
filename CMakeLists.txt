cmake_minimum_required(VERSION 3.16)
project(resnet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create executable
add_executable(resnet ${SOURCES})

# Link libraries
target_link_libraries(resnet ${OpenCV_LIBS})

target_link_libraries(resnet stdc++fs)

# Compiler flags
target_compile_options(resnet PRIVATE -Wall -Wextra -O3)