#include "bicycle_tracking_cost.h"

#include <cmath>
#include <iostream>

namespace {

bool near(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol;
}

bool vector_near(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol = 1e-6) {
    if (a.size() != b.size()) {
        return false;
    }
    return (a - b).cwiseAbs().maxCoeff() <= tol;
}

bool matrix_near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol = 1e-6) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    return (a - b).cwiseAbs().maxCoeff() <= tol;
}

int run_tests() {
    Trajectory ref;
    for (int i = 0; i <= 10; ++i) {
        Eigen::VectorXd s(4);
        s << static_cast<double>(i), 0.0, 0.0, 1.0;
        ref.states.push_back(s);
    }
    ref.approximate_states_with_cubic_spline();

    const double w_cte = 3.0;
    const double w_heading = 4.0;
    const double w_speed = 5.0;
    const double w_cte_terminal = 6.0;
    const double w_heading_terminal = 7.0;
    const double w_speed_terminal = 8.0;
    Eigen::Matrix2d R;
    R << 2.0, 0.5,
         0.5, 5.0;

    BicycleTrackingCost cost(ref, w_cte, w_heading, R, w_cte_terminal, w_heading_terminal, w_speed, w_speed_terminal);

    Eigen::VectorXd x(4);
    x << 5.0, 2.0, 0.1, 4.0;
    Eigen::VectorXd u(2);
    u << 0.3, -0.2;

    const double running = cost.compute_running_cost(x, u);
    const double expected_running = 28.68;
    if (!near(running, expected_running, 1e-6)) {
        std::cerr << "compute_running_cost failed: got " << running << ", want " << expected_running << "\n";
        return 1;
    }

    const double terminal = cost.compute_terminal_cost(x);
    const double expected_terminal = 48.035;
    if (!near(terminal, expected_terminal, 1e-6)) {
        std::cerr << "compute_terminal_cost failed: got " << terminal << ", want " << expected_terminal << "\n";
        return 1;
    }

    Eigen::VectorXd expected_Lx(4);
    expected_Lx << 0.0, 6.0, 0.4, 15.0;
    const Eigen::VectorXd Lx = cost.compute_Lx(x, u);
    if (!vector_near(Lx, expected_Lx, 1e-6)) {
        std::cerr << "compute_Lx failed: got [" << Lx.transpose() << "], want [" << expected_Lx.transpose() << "]\n";
        return 1;
    }

    Eigen::VectorXd expected_Lu(2);
    expected_Lu << 0.5, -0.85;
    const Eigen::VectorXd Lu = cost.compute_Lu(x, u);
    if (!vector_near(Lu, expected_Lu, 1e-6)) {
        std::cerr << "compute_Lu failed: got [" << Lu.transpose() << "], want [" << expected_Lu.transpose() << "]\n";
        return 1;
    }

    Eigen::VectorXd expected_terminal_grad(4);
    expected_terminal_grad << 0.0, 12.0, 0.7, 24.0;
    const Eigen::VectorXd term_grad = cost.terminal_cost_gradient(x);
    if (!vector_near(term_grad, expected_terminal_grad, 1e-6)) {
        std::cerr << "terminal_cost_gradient failed: got [" << term_grad.transpose()
                  << "], want [" << expected_terminal_grad.transpose() << "]\n";
        return 1;
    }

    Eigen::MatrixXd expected_Lxx = Eigen::MatrixXd::Zero(4, 4);
    expected_Lxx(1, 1) = w_cte;
    expected_Lxx(2, 2) = w_heading;
    expected_Lxx(3, 3) = w_speed;
    const Eigen::MatrixXd Lxx = cost.compute_Lxx(x, u);
    if (!matrix_near(Lxx, expected_Lxx, 1e-6)) {
        std::cerr << "compute_Lxx failed.\n";
        return 1;
    }

    const Eigen::MatrixXd Luu = cost.compute_Luu(x, u);
    if (!matrix_near(Luu, R, 1e-6)) {
        std::cerr << "compute_Luu failed.\n";
        return 1;
    }

    const Eigen::MatrixXd Lux = cost.compute_Lux(x, u);
    if (Lux.rows() != 2 || Lux.cols() != 4 || !near(Lux.cwiseAbs().maxCoeff(), 0.0, 1e-12)) {
        std::cerr << "compute_Lux failed.\n";
        return 1;
    }

    Eigen::MatrixXd expected_terminal_hess = Eigen::MatrixXd::Zero(4, 4);
    expected_terminal_hess(1, 1) = w_cte_terminal;
    expected_terminal_hess(2, 2) = w_heading_terminal;
    expected_terminal_hess(3, 3) = w_speed_terminal;
    const Eigen::MatrixXd term_hess = cost.terminal_cost_hessian(x);
    if (!matrix_near(term_hess, expected_terminal_hess, 1e-6)) {
        std::cerr << "terminal_cost_hessian failed.\n";
        return 1;
    }

    const std::string json_cfg = R"json(
    {
      "bicycle_tracking_cost": {
        "w_cte": 1.5,
        "w_heading": 2.5,
        "w_speed": 3.5,
        "w_cte_terminal": 4.5,
        "w_heading_terminal": 5.5,
        "w_speed_terminal": 6.5,
        "R": [[1.0, 0.2], [0.2, 2.0]]
      }
    })json";
    cost.set_weights_from_json_string(json_cfg);

    const double updated_running = cost.compute_running_cost(x, u);
    const double expected_updated_running = 18.8355;
    if (!near(updated_running, expected_updated_running, 1e-6)) {
        std::cerr << "set_weights_from_json_string did not update running cost: got "
                  << updated_running << ", want " << expected_updated_running << "\n";
        return 1;
    }

    Eigen::VectorXd expected_updated_lu(2);
    expected_updated_lu << 0.26, -0.34;
    const Eigen::VectorXd updated_lu = cost.compute_Lu(x, u);
    if (!vector_near(updated_lu, expected_updated_lu, 1e-6)) {
        std::cerr << "set_weights_from_json_string did not update R: got ["
                  << updated_lu.transpose() << "], want [" << expected_updated_lu.transpose() << "]\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main() {
    return run_tests();
}
