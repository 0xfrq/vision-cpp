cmake_minimum_required(VERSION 3.5)
project(vision_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(OpenCV_STATIC ON)
set(BUILD_SHARED_LIBS OFF)

# Find catkin and required ROS packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    v2_detection
)

find_package(OpenCV REQUIRED)

set(ORT_DIR /home/robotis/onnxruntime-linux-x64-1.7.0)

# Catkin package configuration
catkin_package(
    CATKIN_DEPENDS roscpp roslib std_msgs v2_detection
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_DIR}/include
)

link_directories(${ORT_DIR}/lib)

add_executable(vision
    src/main.cpp
    src/yolo_onnx.cpp
)

target_link_libraries(vision
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${ORT_DIR}/lib/libonnxruntime.so.1.7.0
)