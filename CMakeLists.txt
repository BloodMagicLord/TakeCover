cmake_minimum_required(VERSION 3.14)
project(VizDoomBase)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)

include_directories("/headless/base/ViZDoom/include")

add_executable(VizDoomBase main.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(VizDoomBase "/headless/base/ViZDoom/bin/libvizdoom.so" dl)

target_link_libraries( VizDoomBase ${OpenCV_LIBS} )