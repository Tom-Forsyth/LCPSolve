#include <iostream>
#include <Eigen/Dense>
#include "LCPSolve.h"


void printResult(LCP sol, const int &testNum) {
    std::cout << "---Test #" << testNum << "---" << std::endl;
    std::cout << "M = \n" << sol.M << std::endl;
    std::cout << "q = " << sol.q.transpose() << std::endl;
    std::cout << "z = " << sol.z.transpose() << std::endl;
    std::cout << "w = " << sol.w.transpose() << std::endl;
    std::cout << "Exit condition: " << sol.exitCond << std::endl << std::endl;
}

void displayPassRate(const bool &result, int &nTestComplete, int &nTestPassed) {
    nTestComplete++;
    if (result) {
        nTestPassed++;
    }
    double passRate = (nTestPassed/nTestComplete) * 100;
    std::cout << "Pass Rate: " << passRate << " %\n" << std::endl;
}

bool test_1() {
    Eigen::Matrix<double, 2, 2> M {
        {2, -1},
        {-1, 1}
    };
    Eigen::Vector<double, 2> q {-3, 1};
    LCP sol = LCPSolve(M, q);

    Eigen::Vector<double, 2> zTruth {2.0, 1.0};
    double eps = 0.001;
    std::cout << "Running Test 1..." << std::endl;
    if ((sol.z - zTruth).norm() < eps) {
        std::cout << "Test 1 Passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 1 Failed" << std::endl;
        return false;
    }
}

bool test_2() {
    Eigen::Matrix<double, 3, 3> M {
        {2, -1, 3},
        {-1, 1, 6},
        {1, 5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1};
    LCP sol = LCPSolve(M, q);

    Eigen::Vector<double, 3> zTruth {1, 0, 0};
    double eps = 0.001;
    std::cout << "Running Test 2..." << std::endl;
    if ((sol.z - zTruth).norm() < eps) {
        std::cout << "Test 2 Passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 2 Failed" << std::endl;
        return false;
    }
}

bool test_3() {
    Eigen::Matrix<double, 4, 4> M {
            {1, 5, -3, 4},
            {1, -4, -5, -4},
            {-3, 3, -4, 4},
            {4, 9, 0, 1}
        };
    Eigen::Vector<double, 4> q {-3, 4, 5, 1};
    LCP sol = LCPSolve(M, q);
    
    Eigen::Vector<double, 4> zTruth {1.5294, 0.64705, 0.58823, 0};
    double eps = 0.001;
    std::cout << "Running Test 3..." << std::endl;
    if ((sol.z - zTruth).norm() < eps) {
        std::cout << "Test 3 Passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 3 Failed" << std::endl;
        return false;
    }
}

bool test_4() {
    // Secondary ray.
    Eigen::Matrix<double, 3, 3> M {
        {2, -1, 3},
        {-1, 1, 6},
        {1, 5, -2}
    };
    Eigen::Vector<double, 3> q {-2, 5, -1.5};
    LCP sol = LCPSolve(M, q);

    Eigen::Vector<double, 3> zTruth {0, 0, 0.1};
    double eps = 0.001;
    std::cout << "Running Test 4..." << std::endl;
    if ((sol.z - zTruth).norm() < eps) {
        std::cout << "Test 4 Passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 4 Failed" << std::endl;
        return false;
    }
}

bool test_5() {
    Eigen::Matrix<double, 2, 2> M {
        {1, -5},
        {2, 1}
    };
    Eigen::Vector<double, 2> q {-4, 3};
    LCP sol = LCPSolve(M, q);

    Eigen::Vector<double, 2> zTruth {4, 0};
    double eps = 0.001;
    std::cout << "Running Test 5..." << std::endl;
    if ((sol.z - zTruth).norm() < eps) {
        std::cout << "Test 5 Passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 5 Failed" << std::endl;
        return false;
    }
}

int main() {
    // Setup percentage display.
    int nTestComplete = 0;
    int nTestPassed = 0;
    bool result;

    // Run tests.
    std::cout << "Running LCPSolve Tests...\n" << std::endl;

    // Test 1.
    result = test_1();
    displayPassRate(result, nTestComplete, nTestPassed);

    // Test 2.
    result = test_2();
    displayPassRate(result, nTestComplete, nTestPassed);

    // Test 3.
    result = test_3();
    displayPassRate(result, nTestComplete, nTestPassed);

    // Test 4.
    result = test_4();
    displayPassRate(result, nTestComplete, nTestPassed);

    // Test 5.
    result = test_5();
    displayPassRate(result, nTestComplete, nTestPassed);

    return 0;
}