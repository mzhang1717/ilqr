#include "trajectory.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

int run_tests() {
    Trajectory traj;
    for (int i = 0; i <= 10; ++i) {
        Eigen::VectorXd s(4);
        s << static_cast<double>(i), 0.0, 0.0, 1.0;
        traj.states.push_back(s);
    }

    traj.approximate_states_with_cubic_spline();

    Eigen::VectorXd q1(4);
    q1 << 5.0, 2.0, 0.0, 1.0;
    const double cte1 = traj.cross_track_error(q1);
    if (!near(cte1, 2.0, 1e-4)) {
        std::cerr << "cross_track_error positive case failed: got " << cte1 << ", want 2.0\n";
        return 1;
    }

    Eigen::VectorXd q2(4);
    q2 << 5.0, -1.0, 0.0, 1.0;
    const double cte2 = traj.cross_track_error(q2);
    if (!near(cte2, -1.0, 1e-4)) {
        std::cerr << "cross_track_error negative case failed: got " << cte2 << ", want -1.0\n";
        return 1;
    }

    Eigen::VectorXd q3(4);
    q3 << 3.0, 0.2, 0.1, 1.0;
    const double he = traj.heading_error(q3);
    if (!near(he, 0.1, 1e-4)) {
        std::cerr << "heading_error failed: got " << he << ", want 0.1\n";
        return 1;
    }

    const Eigen::VectorXd jac_cte = traj.cross_track_error_jacobian(q1);
    if (jac_cte.size() != 4 || !near(jac_cte(0), 0.0, 1e-4) || !near(jac_cte(1), 1.0, 1e-4) ||
        !near(jac_cte(2), 0.0, 1e-6) || !near(jac_cte(3), 0.0, 1e-6)) {
        std::cerr << "cross_track_error_jacobian failed: got [" << jac_cte.transpose() << "]\n";
        return 1;
    }

    const Eigen::VectorXd jac_he = traj.heading_error_jacobian(q1);
    if (jac_he.size() != 4 || !near(jac_he(0), 0.0, 1e-6) || !near(jac_he(1), 0.0, 1e-6) ||
        !near(jac_he(2), 1.0, 1e-6) || !near(jac_he(3), 0.0, 1e-6)) {
        std::cerr << "heading_error_jacobian failed: got [" << jac_he.transpose() << "]\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main() {
    try {
        return run_tests();
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << '\n';
        return 1;
    }
}
