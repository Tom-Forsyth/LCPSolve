# Project setup.
cmake_minimum_required(VERSION 3.16)
project(LCPSolve
	VERSION 1.0
	DESCRIPTION "LCP solver using Lemke's Method."
	LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 14)

# Default to release build if no build type is selected.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Enable testing.
enable_testing()
add_subdirectory(tests)
add_subdirectory(tests/googletest)

# Add library.
add_library(${PROJECT_NAME} STATIC 
	include/LCPSolve.h
	src/LCPSolve.cpp
)

# Change to the root folder where you have Eigen downloaded.
set(EIGEN_INCLUDE_DIR "C:/src")
target_include_directories(${PROJECT_NAME} PUBLIC 
	${EIGEN_INCLUDE_DIR}
	include
)
