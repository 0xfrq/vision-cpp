cmake_minimum_required(VERSION 3.0.2)
project(vision_cpp)

###################################
# compiler
###################################
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################
# catkin
###################################
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  message_runtime
  v2_detection
)

find_package(OpenCV REQUIRED)

###################################
# onnxruntime (exact .so)
###################################
set(ONNXRUNTIME_DIR /home/robotis/onnxruntime-linux-x64-1.7.0)
set(ONNXRUNTIME_LIB
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.7.0
)

if(NOT EXISTS ${ONNXRUNTIME_LIB})
  message(FATAL_ERROR "libonnxruntime.so.1.7.0 tidak ditemukan")
endif()

###################################
# catkin package
###################################
catkin_package(
  CATKIN_DEPENDS roscpp roslib std_msgs message_runtime v2_detection
)

###################################
# include
###################################
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
)

###################################
# target
###################################
add_executable(vision_yolo_node
  src/main.cpp
  src/yolo_onnx.cpp
)

add_dependencies(vision_yolo_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(vision_yolo_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)

###################################
# rpath (no LD_LIBRARY_PATH needed)
###################################
set_target_properties(vision_yolo_node PROPERTIES
  BUILD_RPATH ${ONNXRUNTIME_DIR}/lib
  INSTALL_RPATH ${ONNXRUNTIME_DIR}/lib
)
