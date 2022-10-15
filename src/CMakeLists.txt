# Source files.
set(SOURCES
	LCPSolve.cpp
)

# Header files.
set(HEADERS
	include/LCPSolve.h
)

# Create library.
add_library(LCPSolveLib STATIC
	${SOURCES}
	${HEADERS}
)

# Include directory.
target_include_directories(LCPSolveLib 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against Eigen.
target_link_libraries(LCPSolveLib
	PUBLIC Eigen3::Eigen
)


