# Project setup.
cmake_minimum_required(VERSION 3.16.3)
project(LCPSolve
	VERSION 1.0
	DESCRIPTION "LCP solver using Lemke's Method."
	HOMEPAGE_URL "https://github.com/Tom-Forsyth/LCPSolve"
	LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 14)

# Setup Eigen3.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/eigen/cmake)
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen)
find_package(Eigen3 REQUIRED)

# Enable testing.
enable_testing()
add_subdirectory(tests)
add_subdirectory(tests/googletest)

# Create LCPSolveLib.
add_subdirectory(src)
