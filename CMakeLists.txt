cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define your executable
add_executable(process process.cpp)

# Link OpenCV libraries
target_link_libraries(process ${OpenCV_LIBS})
