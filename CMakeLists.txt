cmake_minimum_required(VERSION 3.5)
project(vision_cpp)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(OpenCV_STATIC ON)
set(BUILD_SHARED_LIBS OFF)
find_package(OpenCV REQUIRED)
set(ORT_DIR /home/fariq/Desktop/onnxruntime-linux-x64-1.7.0)  # Changed this line
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_DIR}/include
)
link_directories(${ORT_DIR}/lib)
add_executable(vision
    main.cpp
    yolo_onnx.cpp
)
target_link_libraries(vision
    ${OpenCV_LIBS}
    ${ORT_DIR}/lib/libonnxruntime.so.1.7.0

)
