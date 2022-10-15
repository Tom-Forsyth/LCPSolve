#include <gtest/gtest.h>
#include "LCPSolve.h"
#include <limits>

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

TEST(LCPSolveTests, TestNumericalInstablilityInputs)
{
    // Setup.
    Eigen::Matrix<double, 2, 2> M{
        {1, -5},
        {2,  1}
    };
    double badNum = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector<double, 2> q{badNum, -3};

    // Expected.
    int exitConditionExpected = 4;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(sol.exitCond == exitConditionExpected);
}

TEST(LCPSolveTests, TestNumericalInstabilityResults)
{
    // Setup.
    Eigen::Matrix<double, 7, 7> M{
        {0.00596770488864640201442934852594, -0.0260623958248166649742660894162, 0.0369264119309915608835304112745, 0.042503238094155459947387498687, 0.683117626591345894482287803839, 0.587214064854383877012367065618, 1.61815721866978877811868997108e-17},
        {-0.0366206424548746328762049984107, 0.159931112014909143637098054569, -0.226597821725226356015525652765, -0.260819848578926538618105723799, -4.19193087205396786032451927895, -3.60342153554170563367620161443, 1.28773978810819000036490843185e-16},
        {0.00401216066145535302234215180306, -0.0175220660576810086106469555034, 0.0248260763698423050915842225095, 0.0285754445046833734089819500923, 0.459268298231519933594313442882, 0.39479116592115054418243857981, 1.72714731139260471523006871475e-15},
        {0.00295275225856760081821761332321, -0.0128953759563109615626785853237, 0.0182707671147532869915242059733, 0.0210301170417662025569072170583, 0.337998803966067518089744226018, 0.290546816341437641462164265249, 6.31045813269726525369411717622e-17},
        {-0.00532314724884069574023515158956, 0.0232474583146717490855781562686, -0.0329380778285437506025523646258, -0.037912564235233772724775747065, -0.609335713222426633706163556781, -0.523790467547597726749586399819, -3.71962096972654226037105529526e-16},
        {-0.00750447687957157133498142798089, 0.0327738469885936328784836746308, -0.0464355074107086762680296487815, -0.0534484485302463147027296486158, -0.859030486667607728890061480342, -0.738430343873542338606341672858, -6.24721384812335680069956986589e-16},
        {6.63498686914750699476336764437e-18, -4.53557751518716956871193686685e-18, 1.4089833906373785600196422552e-16, -6.0864777209577129987733236765e-17, -2.98456755686358404498096439393e-16, -3.44057955387498226058293876642e-16, -2.85275896559420900654177069533e-16}
    };
    Eigen::Vector<double, 7> q{0.361547229269864822676083804254, 0.287559277362382792464501335417, 0.198421264576616629415894976773, -0.000345374787158553471499988773985, -0.000366872187660661607661038985384, 0.0937967720578135322195834078229, 0.116002210843800926398650119609};

    // Expected.
    int exitConditionExpected = 4;

    // Test solution.
    LCPSolve::LCP sol = LCPSolve::LCPSolve(M, q);
    EXPECT_TRUE(sol.exitCond == exitConditionExpected);
}
