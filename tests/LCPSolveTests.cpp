#include <gtest/gtest.h>
#include "LCPSolve.h"

// Test 1.
TEST(LCPSolveTests, Test1)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M {
        {2, -1},
        {-1, 1}
    };
    Eigen::Vector<double, 2> q {-3, 1};
    Eigen::Vector<double, 2> zTruth {2.0, 1.0};

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    double eps = 1e-10;
    bool pass = false;
    if ((sol.z - zTruth).norm() < eps) {
        pass = true;
    }
    EXPECT_TRUE(pass);
}

// Test 2.
TEST(LCPSolveTests, Test2)
{
    // Setup.
    Eigen::Matrix<double, 3, 3> M {
        { 2, -1,  3},
        {-1,  1,  6},
        { 1,  5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1};
    Eigen::Vector<double, 3> zTruth {1, 0, 0};

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    double eps = 1e-10;
    bool pass = false;
    if ((sol.z - zTruth).norm() < eps) {
        pass = true;
    }
    EXPECT_TRUE(pass);
}

// Test 3.
TEST(LCPSolveTests, Test3)
{
    // Setup.
    Eigen::Matrix<double, 4, 4> M {
        { 1,  5, -3,  4},
        { 1, -4, -5, -4},
        {-3,  3, -4,  4},
        { 4,  9,  0,  1}
    };
    Eigen::Vector<double, 4> q {-3, 4, 5, 1};
    Eigen::Vector<double, 4> zTruth {1.5294, 0.64705, 0.58823, 0};

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    double eps = 1e-3; 
    bool pass = false;
    if ((sol.z - zTruth).norm() < eps) {
        pass = true;
    }
    EXPECT_TRUE(pass);
}

// Test 4 (Secondary Ray).
TEST(LCPSolveTests, Test4)
{
    // Setup.
    Eigen::Matrix<double, 3, 3> M {
        { 2, -1,  3},
        {-1,  1,  6},
        { 1,  5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1.5};
    Eigen::Vector<double, 3> zTruth {0, 0, 0.1};

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    double eps = 1e-10;
    bool pass = false;
    if ((sol.z - zTruth).norm() < eps) {
        pass = true;
    }
    EXPECT_TRUE(pass);
}

// Test 5.
TEST(LCPSolveTests, Test5)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M {
        {1, -5},
        {2,  1}
    };
    Eigen::Vector<double, 2> q {-4, 3};
    Eigen::Vector<double, 2> zTruth {4, 0};

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    double eps = 1e-10;
    bool pass = false;
    if ((sol.z - zTruth).norm() < eps) {
        pass = true;
    }
    EXPECT_TRUE(pass);
}
