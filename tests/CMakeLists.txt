# Setup.
cmake_minimum_required(VERSION 3.16)
set(THIS LCPSolveTests)

# Create test executable.
add_executable(${THIS}
    LCPSolveTests.cpp
)
target_link_libraries(${THIS} PUBLIC
    gtest_main
    LCPSolve
)

# Add tests.
add_test(
    NAME ${THIS}
    COMMAND ${THIS}
)
