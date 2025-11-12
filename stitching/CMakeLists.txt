cmake_minimum_required(VERSION 3.16)
project(image-stitching)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS_DEBUG "-g -Wall")

if(DEFINED ENV{OpenCV_DIR})
    # Allow callers to override the OpenCV install directory via env var.
    set(OpenCV_DIR $ENV{OpenCV_DIR})
endif()

find_package(OpenCV 4 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        include
)

add_executable(${PROJECT_NAME}
        src/app.cc include/app.h
        src/sensor_data_interface.cc include/sensor_data_interface.h
        src/stitching_param_generator.cc include/stitching_param_generator.h
        src/image_stitcher.cc include/image_stitcher.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -lpthread -lm)
