cmake_minimum_required(VERSION 3.21)
project(C___code)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(C___code
        main.cpp
        method.cpp
        method.h)
