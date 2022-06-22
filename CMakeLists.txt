cmake_minimum_required(VERSION 3.16)

project(LCPSolve
	VERSION 1.0
	DESCRIPTION "LCP solver using Lemke's Method."
	LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O1")

include_directories(include)

add_executable(tests
	tests/tests.cpp
	src/LCPSolve.cpp
	include/LCPSolve.h
)
