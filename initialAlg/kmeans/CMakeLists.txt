cmake_minimum_required(VERSION 2.8)
project( kmeans )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( kmeans kmeans.cpp )
target_link_libraries( kmeans ${OpenCV_LIBS} )
