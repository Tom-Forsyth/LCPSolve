# Project setup.
cmake_minimum_required(VERSION 3.16.3)
project(LCPSolve
	VERSION 1.0
	DESCRIPTION "LCP solver using Lemke's Method."
	HOMEPAGE_URL "https://github.com/Tom-Forsyth/LCPSolve"
	LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 14)

# Use release build.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Setup Eigen3.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/eigen/cmake)
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen)
find_package(Eigen3 REQUIRED)

# Setup Google Test.
add_subdirectory(external/googletest)
enable_testing()

add_subdirectory(tests)

# Create LCPSolveLib.
add_subdirectory(src)
