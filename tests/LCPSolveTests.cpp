#include <gtest/gtest.h>
#include "LCPSolve.h"

// Helper function to compare Eigen vectors.
bool isApproxEqual(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2, double tolerance = 1e-10)
{
    if ((vec1 - vec2).norm() < tolerance) {
        return true;
    }
    return false;
}

TEST(LCPSolveTests, R2Test)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M {
        {2, -1},
        {-1, 1}
    };
    Eigen::Vector<double, 2> q {-3, 1};

    // Expected.
    Eigen::Vector<double, 2> zTruth {2.0, 1.0};
    int exitConditionExpected = 0;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(isApproxEqual(sol.z, zTruth) && sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, R3Test)
{
    // Setup.
    Eigen::Matrix<double, 3, 3> M {
        { 2, -1,  3},
        {-1,  1,  6},
        { 1,  5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1};

    // Expected.
    Eigen::Vector<double, 3> zTruth {1, 0, 0};
    int exitConditionExpected = 0;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(isApproxEqual(sol.z, zTruth) && sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, R4Test)
{
    // Setup.
    Eigen::Matrix<double, 4, 4> M {
        { 1,  5, -3,  4},
        { 1, -4, -5, -4},
        {-3,  3, -4,  4},
        { 4,  9,  0,  1}
    };
    Eigen::Vector<double, 4> q {-3, 4, 5, 1};

    // Expected.
    Eigen::Vector<double, 4> zTruth {1.5294, 0.64705, 0.58823, 0};
    int exitConditionExpected = 0;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(isApproxEqual(sol.z, zTruth, 1e-3) && sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, R3SecondaryRayTest)
{
    // Setup.
    Eigen::Matrix<double, 3, 3> M {
        { 2, -1,  3},
        {-1,  1,  6},
        { 1,  5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1.5};

    // Expected.
    Eigen::Vector<double, 3> zTruth {0, 0, 0.1};
    int exitConditionExpected = 1;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(isApproxEqual(sol.z, zTruth) && sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, R2Test2)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M {
        {1, -5},
        {2,  1}
    };
    Eigen::Vector<double, 2> q {-4, 3};

    // Expected.
    Eigen::Vector<double, 2> zTruth {4, 0};
    int exitConditionExpected = 0;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(isApproxEqual(sol.z, zTruth) && sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, TestIncorrectInputDims)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M{
        {1, -5},
        {2,  1}
    };
    Eigen::Vector<double, 3> q{ -4, 3, -3};

    // Expected.
    int exitConditionExpected = 2;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(sol.exitCond == exitConditionExpected);
}
