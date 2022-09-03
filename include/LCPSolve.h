#pragma once

#include <Eigen/Dense>

namespace LCPSolve
{
    // Linear complementarity problem datatype.
    struct LCP { 
        int exitCond;
        Eigen::VectorXd w {};
        Eigen::VectorXd z {};
        Eigen::MatrixXd M {};
        Eigen::VectorXd q {};
    };

    // LCP Solver.
    LCP LCPSolve(Eigen::MatrixXd M, Eigen::VectorXd q);
}
