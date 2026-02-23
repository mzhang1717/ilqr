#include "bicycle_model.h"

#include <cmath>
#include <iostream>

namespace {

bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

bool matrix_near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol = 1e-5) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    return (a - b).cwiseAbs().maxCoeff() <= tol;
}

int run_tests() {
    const double wheelbase = 2.5;
    const double dt = 0.1;
    BicycleModel model(wheelbase, dt);
    Eigen::VectorXd lower(2);
    lower << -1.0, -0.4;
    Eigen::VectorXd upper(2);
    upper << 2.0, 0.4;
    BicycleModel model_with_limits(wheelbase, dt, lower, upper);

    if (!matrix_near(model_with_limits.control_lower_bound_, lower, 1e-12) ||
        !matrix_near(model_with_limits.control_upper_bound_, upper, 1e-12)) {
        std::cerr << "constructor control limits were not set correctly.\n";
        return 1;
    }

    Eigen::VectorXd updated_lower(2);
    updated_lower << -2.0, -0.2;
    Eigen::VectorXd updated_upper(2);
    updated_upper << 1.5, 0.2;
    model.set_control_limits(updated_lower, updated_upper);
    if (!matrix_near(model.control_lower_bound_, updated_lower, 1e-12) ||
        !matrix_near(model.control_upper_bound_, updated_upper, 1e-12)) {
        std::cerr << "set_control_limits did not update limits correctly.\n";
        return 1;
    }

    const std::string json_cfg = R"json(
    {
      "bicycle_model": {
        "wheel_base": 3.1,
        "control_lower_bound": [-3.5, -0.61],
        "control_upper_bound": [3.5, 0.61]
      }
    })json";
    model.set_from_json_string(json_cfg);
    if (!near(model.wheelbase(), 3.1, 1e-12)) {
        std::cerr << "set_from_json_string did not update wheelbase.\n";
        return 1;
    }
    Eigen::VectorXd json_lower(2);
    json_lower << -3.5, -0.61;
    Eigen::VectorXd json_upper(2);
    json_upper << 3.5, 0.61;
    if (!matrix_near(model.control_lower_bound_, json_lower, 1e-12) ||
        !matrix_near(model.control_upper_bound_, json_upper, 1e-12)) {
        std::cerr << "set_from_json_string did not update control bounds.\n";
        return 1;
    }
    model.set_dt(0.2);
    if (!near(model.dt(), 0.2, 1e-12)) {
        std::cerr << "set_dt did not update dt.\n";
        return 1;
    }

    Eigen::VectorXd x(4);
    x << 1.0, 2.0, 0.3, 4.0;
    Eigen::VectorXd u(2);
    u << 0.5, 0.1;

    const Eigen::VectorXd next = model.dynamics(x, u);

    const double expected_x = x(0) + model.dt() * x(3) * std::cos(x(2));
    const double expected_y = x(1) + model.dt() * x(3) * std::sin(x(2));
    const double expected_yaw = x(2) + model.dt() * x(3) / model.wheelbase() * std::tan(u(1));
    const double expected_v = x(3) + model.dt() * u(0);

    if (!near(next(0), expected_x) || !near(next(1), expected_y) ||
        !near(next(2), expected_yaw) || !near(next(3), expected_v)) {
        std::cerr << "dynamics mismatch: got [" << next.transpose() << "]\n";
        return 1;
    }

    const Eigen::MatrixXd A = model.state_jacobian(x, u);
    const Eigen::MatrixXd B = model.control_jacobian(x, u);

    const double eps = 1e-6;
    Eigen::MatrixXd A_fd(4, 4);
    Eigen::MatrixXd B_fd(4, 2);

    for (int i = 0; i < 4; ++i) {
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus(i) += eps;
        x_minus(i) -= eps;
        A_fd.col(i) = (model.dynamics(x_plus, u) - model.dynamics(x_minus, u)) / (2.0 * eps);
    }

    for (int i = 0; i < 2; ++i) {
        Eigen::VectorXd u_plus = u;
        Eigen::VectorXd u_minus = u;
        u_plus(i) += eps;
        u_minus(i) -= eps;
        B_fd.col(i) = (model.dynamics(x, u_plus) - model.dynamics(x, u_minus)) / (2.0 * eps);
    }

    if (!matrix_near(A, A_fd, 2e-5)) {
        std::cerr << "state_jacobian mismatch.\nA:\n" << A << "\nA_fd:\n" << A_fd << "\n";
        return 1;
    }

    if (!matrix_near(B, B_fd, 2e-5)) {
        std::cerr << "control_jacobian mismatch.\nB:\n" << B << "\nB_fd:\n" << B_fd << "\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main() {
    return run_tests();
}
