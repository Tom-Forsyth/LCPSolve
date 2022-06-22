# LCPSolve
An LCP solver based on Lemke's method.

## Problem Statement.
Given a matrix $M$ and a vector $q$, solve the LCP below for $z$ and $w$.

$Mz + q = w$
$0 \leq z \perp w \geq 0$

## Example Code
    Eigen::Matrix<double, 2, 2> M {
        {1, -5},
        {2, 1}
    };
    Eigen::Vector<double, 2> q {-4, 3};
    LCP sol = LCPSolve(M, q);

## Output
    sol.z = [4 0]
    sol.w = [0 11]
    sol.exitCond = 0

## Exit Conditions
    0 - Solution found.
    1 - Secondary ray termination.
    2 - Incorrect input dimensions.
    3 - Exceeded maximum iterations.
